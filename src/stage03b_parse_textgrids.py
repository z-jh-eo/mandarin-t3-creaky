"""
Stage 3b — parse MFA TextGrids, extract T3 rime intervals.
Input:  transcripts_tone.parquet + mfa_output/ TextGrids
Output: t3_syllables.parquet (one row per T3 syllable, rime-level intervals)
Run:    python -m src.stage03b_parse_textgrids --config config.yaml
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Minimal TextGrid parser (no external dependency)
# Handles MFA's long_textgrid format.
# ---------------------------------------------------------------------------
def parse_textgrid(path: Path) -> dict[str, list[tuple[float, float, str]]]:
    """Return {tier_name: [(start, end, label), ...]} for interval tiers."""
    text = path.read_text(encoding="utf-8")
    tiers = {}
    # Split on tier blocks
    for block in re.split(r"item\s*\[\d+\]", text)[1:]:
        name_m = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_m:
            continue
        name = name_m.group(1)
        intervals = re.findall(
            r'xmin\s*=\s*([0-9.]+)\s+xmax\s*=\s*([0-9.]+)\s+text\s*=\s*"([^"]*)"', block
        )
        if intervals:
            tiers[name] = [(float(a), float(b), c) for a, b, c in intervals]
    return tiers


# ---------------------------------------------------------------------------
# Rime boundary detection
# A syllable's phones: [initial?] + [medial?] + nucleus + [coda?]
# Rime = everything after the initial consonant phone.
# MFA mandarin_china_mfa initials: b p m f d t n l g k h j q x
#   zh ch sh r z c s y w (and ng is coda only, never initial)
# ---------------------------------------------------------------------------
"""
INITIALS = {
    "b","p","m","f","d","t","n","l","g","k","h",
    "j","q","x","zh","ch","sh","r","z","c","s","y","w",
}
"""

INITIALS = {
    "p",
    "pʰ",
    "b",
    "m",
    "f",
    "t",
    "tʰ",
    "d",
    "n",
    "l",
    "ʂ",
    "ʃ",
    "ʒ",
    "ts",
    "tsʰ",
    "dz",
    "s",
    "z",
    "ɕ",
    "ʑ",
    "tɕ",
    "tɕʰ",
    "dʑ",
    "ɲ",
    "ɣ",
    "k",
    "kʰ",
    "g",
    "x",
    "h",
    "ɥ",
    "w",
    "j",
}


def rime_bounds(
    phones: list[tuple[float, float, str]], max_phone_ms: float = 500.0
) -> tuple[float, float] | None:
    """Given phone intervals for one syllable, return (rime_start, rime_end).
    Rejects any chunk where a single phone exceeds max_phone_ms (alignment failure).
    """
    if not phones:
        return None
    phones = [
        (s, e, lab) for s, e, lab in phones if lab.strip() not in ("", "sp", "sil")
    ]
    if not phones:
        return None
    # Reject if any individual phone is suspiciously long (alignment failure)
    if any((e - s) * 1000 > max_phone_ms for s, e, lab in phones):
        return None
    first_lab = re.sub(r"\d", "", phones[0][2]).lower()
    if first_lab in INITIALS and len(phones) > 1:
        rime_start = phones[1][0]
    else:
        rime_start = phones[0][0]
    rime_end = phones[-1][1]
    return rime_start, rime_end


# ---------------------------------------------------------------------------
# Match transcript syllables to TextGrid phone intervals
# Strategy: for each syllable in the transcript (ordered by turn start),
# find the phone cluster in the TextGrid whose midpoint falls within
# [turn_start, turn_end] and whose onset is closest to expected order.
# We group phones into syllables by matching to pypinyin finals.
# ---------------------------------------------------------------------------
def phones_in_window(
    phones_tier: list[tuple[float, float, str]], t_start: float, t_end: float
) -> list[tuple[float, float, str]]:
    """All phones whose midpoint is within [t_start, t_end]."""
    return [
        (s, e, lab)
        for s, e, lab in phones_tier
        if t_start <= (s + e) / 2 <= t_end and lab.strip() not in ("", "sp", "sil")
    ]


def assign_phones_to_syllables(
    turn_syls: pd.DataFrame,
    phones_tier: list[tuple[float, float, str]],
    turn_start: float,
    turn_end: float,
) -> list[dict]:
    """
    Assign phone intervals to syllables within one speaker turn.
    Uses time interpolation: each syllable gets an equal time window
    within the turn, and phones whose midpoint falls in that window
    are assigned to it.
    Returns list of dicts with rime_start, rime_end, rime_duration_ms.
    """
    turn_phones = sorted(
        phones_in_window(phones_tier, turn_start, turn_end), key=lambda x: x[0]
    )
    n_syls = len(turn_syls)
    if n_syls == 0 or not turn_phones:
        return []

    turn_dur = turn_end - turn_start
    results = []
    for i, syl_row in enumerate(turn_syls.itertuples()):
        # Estimated time window for this syllable position
        syl_t0 = turn_start + i * turn_dur / n_syls
        syl_t1 = turn_start + (i + 1) * turn_dur / n_syls
        chunk = [
            (s, e, lab) for s, e, lab in turn_phones if syl_t0 <= (s + e) / 2 <= syl_t1
        ]
        bounds = rime_bounds(chunk)
        if bounds is None:
            continue
        r_start, r_end = bounds
        dur_ms = (r_end - r_start) * 1000
        results.append(
            {
                **syl_row._asdict(),
                "rime_start": r_start,
                "rime_end": r_end,
                "rime_duration_ms": dur_ms,
            }
        )
    return results


# ---------------------------------------------------------------------------
def process_file(
    file_id: str,
    speaker_id: str,
    audio_path: str,
    tg_path: Path,
    tokens: pd.DataFrame,
    min_rime_ms: float,
    max_phone_ms: float,
) -> list[dict]:
    """Extract T3 rime rows for one speaker in one file."""
    try:
        tiers = parse_textgrid(tg_path)
    except Exception as e:
        warnings.warn(f"TextGrid parse failed {tg_path}: {e}")
        return []

    # MFA names the phone tier after the speaker or 'phones'
    phone_tier = None
    for name in tiers:
        if "phone" in name.lower() or name == speaker_id:
            phone_tier = tiers[name]
            break
    if phone_tier is None and tiers:
        phone_tier = list(tiers.values())[0]  # fallback: first tier
    if phone_tier is None:
        return []

    spkr_t3 = tokens[
        (tokens["file_id"] == file_id)
        & (tokens["speaker_id"] == speaker_id)
        & (tokens["underlying_tone"] == 3)
    ]
    if spkr_t3.empty:
        return []

    rows = []
    for turn_idx, turn_df in spkr_t3.groupby("turn_idx"):
        turn_start = turn_df["turn_start"].iloc[0]
        turn_end = turn_df["turn_end"].iloc[0]
        assigned = assign_phones_to_syllables(turn_df, phone_tier, turn_start, turn_end)
        for r in assigned:
            if r["rime_duration_ms"] < min_rime_ms:
                continue
            r["audio_path"] = audio_path
            rows.append(r)
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mfa-output", default="data/interim/mfa_output")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    excl = cfg["exclusions"]
    min_rime_ms = excl["min_rime_duration_ms"]
    max_ph_ms = excl["alignment_phone_max_duration_ms"]

    tokens = pd.read_parquet(args.tokens)
    manifest = pd.read_parquet(args.manifest)
    corpus_root = Path(cfg["paths"]["corpus_root"])
    mfa_output = Path(args.mfa_output)

    all_rows = []
    n_missing_tg = 0

    for _, file_row in tqdm(manifest.iterrows(), total=len(manifest), desc="Stage 3b"):
        file_id = file_row["file_id"]
        audio_path = str(corpus_root / file_row["audio_path"])

        for spkr in file_row["chosen_speakers"]:
            tg_path = mfa_output / spkr / f"{file_id}.TextGrid"
            if not tg_path.exists():
                n_missing_tg += 1
                warnings.warn(f"Missing TextGrid: {tg_path}")
                continue
            all_rows.extend(
                process_file(
                    file_id, spkr, audio_path, tg_path, tokens, min_rime_ms, max_ph_ms
                )
            )

    df = pd.DataFrame(all_rows)
    # Drop index artifacts from itertuples
    df = df.drop(columns=["Index"], errors="ignore")

    print(f"\nT3 syllables extracted:  {len(df):,}")
    print(f"Missing TextGrids:       {n_missing_tg}")
    if not df.empty:
        print(
            f"Rime duration (ms):      "
            f"mean={df['rime_duration_ms'].mean():.1f}, "
            f"median={df['rime_duration_ms'].median():.1f}"
        )
        print(f"Sandhi breakdown:\n{df['sandhi_context'].value_counts().to_string()}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


def load_config(path):
    return yaml.safe_load(open(path, encoding="utf-8"))


if __name__ == "__main__":
    main()
