"""
Stage 4 — prosodic position coding.
Detects IPUs via energy-based RMS silence, assigns each T3 syllable
a prosodic position (initial/medial/final) and continuous position.
Input:  t3_syllables.parquet + audio files
Output: t3_prosody.parquet
Run:    python -m src.stage04_prosody --config config.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


def load_config(path: str) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------------------
# IPU detection via RMS energy
# ---------------------------------------------------------------------------
def detect_ipus(
    audio_path: str, silence_thresh_ms: float, sr: int = 16000, frame_ms: int = 10
) -> list[tuple[float, float]]:
    """
    Return list of (ipu_start, ipu_end) in seconds.
    Silence = frames where RMS < mean_rms * 0.1.
    """
    y, _ = librosa.load(audio_path, sr=sr, mono=True)
    frame_len = int(sr * frame_ms / 1000)
    hop_len = frame_len

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    threshold = rms.mean() * 0.1
    is_speech = rms > threshold

    # Convert frame-level is_speech to time segments
    silence_frames = int(silence_thresh_ms / frame_ms)
    ipus = []
    in_ipu = False
    ipu_start = 0
    silence_count = 0

    for i, speech in enumerate(is_speech):
        t = i * hop_len / sr
        if speech:
            if not in_ipu:
                ipu_start = t
                in_ipu = True
            silence_count = 0
        else:
            if in_ipu:
                silence_count += 1
                if silence_count >= silence_frames:
                    ipu_end = (i - silence_count) * hop_len / sr
                    if ipu_end > ipu_start:
                        ipus.append((ipu_start, ipu_end))
                    in_ipu = False
                    silence_count = 0

    if in_ipu:
        ipus.append((ipu_start, len(y) / sr))

    return ipus


# ---------------------------------------------------------------------------
# Assign T3 syllables to IPUs
# ---------------------------------------------------------------------------
def assign_prosody(
    t3_file: pd.DataFrame, ipus: list[tuple[float, float]], audio_dur: float
) -> pd.DataFrame:
    """
    For each T3 syllable, find its IPU by rime midpoint, then compute
    positional features.
    """
    rows = []
    ipu_arr = np.array(ipus)  # shape (N, 2)

    # Precompute pause_following: silence between consecutive IPUs
    pause_after = {}  # ipu_idx -> pause_ms
    for i in range(len(ipus) - 1):
        pause_after[i] = (ipus[i + 1][0] - ipus[i][1]) * 1000
    pause_after[len(ipus) - 1] = 0.0  # last IPU: no following pause

    for _, row in t3_file.iterrows():
        mid = (row["rime_start"] + row["rime_end"]) / 2

        # Find IPU whose window contains mid
        ipu_idx = None
        for i, (s, e) in enumerate(ipus):
            if s <= mid <= e:
                ipu_idx = i
                break

        if ipu_idx is None:
            # Fallback: nearest IPU by midpoint distance
            ipu_mids = (ipu_arr[:, 0] + ipu_arr[:, 1]) / 2
            ipu_idx = int(np.argmin(np.abs(ipu_mids - mid)))

        ipu_start, ipu_end = ipus[ipu_idx]
        ipu_dur_ms = (ipu_end - ipu_start) * 1000

        # Continuous position within IPU duration
        pos_cont = (mid - ipu_start) / max(ipu_end - ipu_start, 1e-6)
        pos_cont = float(np.clip(pos_cont, 0.0, 1.0))

        rows.append(
            {
                **row.to_dict(),
                "ipu_idx": ipu_idx,
                "ipu_duration_ms": ipu_dur_ms,
                "pos_in_ipu_continuous": pos_cont,
                "pause_following_ms": pause_after[ipu_idx],
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Syllable position within each IPU (by rime_start order)
    df = df.sort_values(["ipu_idx", "rime_start"])
    df["pos_in_ipu"] = df.groupby("ipu_idx").cumcount()
    df["ipu_length_syls"] = df.groupby("ipu_idx")["pos_in_ipu"].transform("count")

    # Categorical position
    def cat_pos(r):
        if r["ipu_length_syls"] == 1:
            return "final"  # single-syllable IPU → treat as final
        if r["pos_in_ipu"] == 0:
            return "initial"
        if r["pos_in_ipu"] == r["ipu_length_syls"] - 1:
            return "final"
        return "medial"

    df["prosodic_position"] = df.apply(cat_pos, axis=1)
    return df


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    corpus_root = Path(cfg["paths"]["corpus_root"])
    sil_thresh = cfg["prosody"]["ipu_silence_threshold_ms"]

    t3 = pd.read_parquet(args.input)
    all_dfs = []

    for file_id, file_df in tqdm(t3.groupby("file_id"), desc="Stage 4"):
        audio_path = str(file_df["audio_path"].iloc[0])
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        audio_dur = len(y) / sr

        ipus = detect_ipus(audio_path, sil_thresh)
        if not ipus:
            continue

        result = assign_prosody(file_df, ipus, audio_dur)
        if not result.empty:
            all_dfs.append(result)

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\nT3 syllables with prosody:  {len(df):,}")
    print(
        f"Prosodic position breakdown:\n{df['prosodic_position'].value_counts().to_string()}"
    )
    print(
        f"Pause following (ms):  mean={df['pause_following_ms'].mean():.1f}, "
        f"median={df['pause_following_ms'].median():.1f}"
    )
    print(
        f"Continuous position:   mean={df['pos_in_ipu_continuous'].mean():.2f}, "
        f"std={df['pos_in_ipu_continuous'].std():.2f}"
    )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
