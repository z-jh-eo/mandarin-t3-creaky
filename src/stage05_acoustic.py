"""
Stage 5 — voice quality feature extraction.
Extracts F0, H1*-H2*, SHR, HNR (0-500 Hz), CPP per T3 rime.
Input:  t3_prosody.parquet + audio files
Output: t3_acoustic.parquet
Run:    python -m src.stage05_acoustic --config config.yaml
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import parselmouth
import yaml
from parselmouth.praat import call
from tqdm import tqdm


def load_config(path: str) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------
def extract_f0(snd: parselmouth.Sound, t0: float, t1: float) -> np.ndarray:
    """Voiced F0 frames (Hz) within [t0, t1]. Unvoiced frames excluded."""
    pitch = snd.to_pitch(time_step=0.005, pitch_floor=50, pitch_ceiling=400)
    frames = [pitch.get_value_at_time(t) for t in np.arange(t0, t1, 0.005)]
    vals = np.array([f for f in frames if f and not np.isnan(f)])
    return vals


def extract_hnr(snd: parselmouth.Sound, t0: float, t1: float) -> np.ndarray:
    """HNR 0-500 Hz frames (dB) within [t0, t1]."""
    seg = snd.extract_part(t0, t1, preserve_times=True)
    try:
        harm = call(seg, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        frames = []
        for t in np.arange(t0, t1, 0.005):
            v = call(harm, "Get value at time", t, "Nearest")
            if v and not np.isnan(v) and v > -200:
                frames.append(v)
        return np.array(frames)
    except Exception:
        return np.array([])


def extract_cpp(snd: parselmouth.Sound, t0: float, t1: float) -> np.ndarray:
    """CPP (dB) within [t0, t1] via LTAS-based cepstrum."""
    seg = snd.extract_part(t0, t1, preserve_times=True)
    try:
        cpp = call(seg, "To PowerCepstrogram", 60, 0.002, 5000, 50)
        trend = call(cpp, "To PowerCepstrum (slice)", (t0 + t1) / 2)
        val = call(
            trend,
            "Get peak prominence (dB)",
            60,
            333,
            "Parabolic",
            0.001,
            0.05,
            "Straight",
            "Robust",
        )
        return np.array([val]) if val and not np.isnan(val) else np.array([])
    except Exception:
        return np.array([])


def extract_h1h2(
    snd: parselmouth.Sound, t0: float, t1: float, f0_mean: float
) -> np.ndarray:
    """H1*-H2* (dB) approximated via spectrum at rime midpoint."""
    if np.isnan(f0_mean) or f0_mean <= 0:
        return np.array([])
    seg = snd.extract_part(t0, t1, preserve_times=True)
    try:
        spec = seg.to_spectrum()
        h1 = call(
            spec, "Get sound pressure level of nearest maximum", f0_mean, "Sinc70"
        )
        h2 = call(
            spec, "Get sound pressure level of nearest maximum", 2 * f0_mean, "Sinc70"
        )
        val = h1 - h2
        return np.array([val]) if not np.isnan(val) else np.array([])
    except Exception:
        return np.array([])


def extract_shr(
    snd: parselmouth.Sound, t0: float, t1: float, f0_mean: float
) -> np.ndarray:
    """SHR approximated as energy at f0/2 relative to f0 (subharmonic check)."""
    if np.isnan(f0_mean) or f0_mean <= 0 or f0_mean / 2 < 50:
        return np.array([])
    seg = snd.extract_part(t0, t1, preserve_times=True)
    try:
        spec = seg.to_spectrum()
        sub = call(
            spec, "Get sound pressure level of nearest maximum", f0_mean / 2, "Sinc70"
        )
        h1 = call(
            spec, "Get sound pressure level of nearest maximum", f0_mean, "Sinc70"
        )
        val = sub - h1
        return np.array([val]) if not np.isnan(val) else np.array([])
    except Exception:
        return np.array([])


# ---------------------------------------------------------------------------
def stats(arr: np.ndarray) -> tuple[float, float, float, float]:
    """Return (mean, std, min, max) or NaNs if empty."""
    if len(arr) == 0:
        return np.nan, np.nan, np.nan, np.nan
    return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())


def extract_features(snd: parselmouth.Sound, t0: float, t1: float) -> dict:
    """Extract all voice quality measures for one rime interval."""
    f0_vals = extract_f0(snd, t0, t1)
    f0_mean = float(f0_vals.mean()) if len(f0_vals) > 0 else np.nan

    hnr_vals = extract_hnr(snd, t0, t1)
    cpp_vals = extract_cpp(snd, t0, t1)
    h1h2_vals = extract_h1h2(snd, t0, t1, f0_mean)
    shr_vals = extract_shr(snd, t0, t1, f0_mean)

    result = {}
    for prefix, vals in [
        ("f0", f0_vals),
        ("hnr500", hnr_vals),
        ("cpp", cpp_vals),
        ("h1h2", h1h2_vals),
        ("shr", shr_vals),
    ]:
        m, s, mn, mx = stats(vals)
        result[f"{prefix}_mean"] = m
        result[f"{prefix}_std"] = s
        result[f"{prefix}_min"] = mn
        result[f"{prefix}_max"] = mx
    return result


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus_root = Path(cfg["paths"]["corpus_root"])
    t3 = pd.read_parquet(args.input)

    all_rows = []
    n_failed = 0

    for file_id, file_df in tqdm(t3.groupby("file_id"), desc="Stage 5"):
        audio_path = str(file_df["audio_path"].iloc[0])
        try:
            snd = parselmouth.Sound(audio_path)
        except Exception as e:
            warnings.warn(f"Could not load {audio_path}: {e}")
            n_failed += len(file_df)
            continue

        for _, row in file_df.iterrows():
            t0 = row["rime_start"]
            t1 = row["rime_end"]
            # Skip rimes too short for reliable extraction
            if (t1 - t0) < 0.03:
                n_failed += 1
                continue
            feats = extract_features(snd, t0, t1)
            all_rows.append({**row.to_dict(), **feats})

    df = pd.DataFrame(all_rows)
    print(f"\nT3 syllables with acoustics: {len(df):,}")
    print(f"Failed/skipped:              {n_failed:,}")
    for m in ["f0", "h1h2", "shr", "hnr500", "cpp"]:
        col = f"{m}_mean"
        valid = df[col].notna().sum()
        print(
            f"  {col}: {valid:,} valid, "
            f"mean={df[col].mean():.2f}, std={df[col].std():.2f}"
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
