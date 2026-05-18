"""
Stage 2 — transcript parsing, pinyin conversion, sandhi tagging.
Input:  manifest_subset.parquet + TXT files
Output: transcripts_tone.parquet (one row per syllable)
Run:    python -m src.stage02_transcripts --config config.yaml
          --manifest data/interim/manifest_subset.parquet
          --output data/interim/transcripts_tone.parquet
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import jieba
import pandas as pd
import yaml
from pypinyin import Style, lazy_pinyin
from tqdm import tqdm

# ---------------------------------------------------------------------------
# TXT parsing
# ---------------------------------------------------------------------------
LINE_RE = re.compile(r"\[([0-9.]+),([0-9.]+)\]\t(\S+)\t(\S+)\t(.+)")


def parse_txt(path: Path) -> list[dict]:
    """Return list of turn dicts from one TXT file."""
    turns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        m = LINE_RE.match(line.strip())
        if not m:
            continue
        start, end, spk, meta, text = m.groups()
        gender = meta.split(",")[0]  # 男 / 女
        turns.append(
            dict(
                start=float(start),
                end=float(end),
                speaker_id=spk,
                gender=gender,
                text=text.strip(),
            )
        )
    return turns


# ---------------------------------------------------------------------------
# Pinyin helpers
# ---------------------------------------------------------------------------
INITIALS = {
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "zh",
    "ch",
    "sh",
    "r",
    "z",
    "c",
    "s",
    "y",
    "w",
}


def split_initial_final(syl: str) -> tuple[str, str]:
    """Split a bare pinyin syllable (no tone digit) into initial + final."""
    for init in sorted(INITIALS, key=len, reverse=True):
        if syl.startswith(init):
            return init, syl[len(init) :]
    return "", syl  # no initial (e.g. 'an', 'er')


def syllable_rows(char: str, pinyin_tone: str, word_idx: int, syl_idx: int) -> dict:
    """Build a partial row for one syllable."""
    # pypinyin returns e.g. 'zhong1'; tone digit is last char
    if pinyin_tone and pinyin_tone[-1].isdigit():
        tone = int(pinyin_tone[-1])
        bare = pinyin_tone[:-1]
    else:
        tone = 0  # neutral / unknown
        bare = pinyin_tone
    init, final = split_initial_final(bare)
    return dict(
        char=char,
        pinyin=pinyin_tone,
        initial=init,
        final=final,
        underlying_tone=tone,
        word_idx=word_idx,
        syl_idx=syl_idx,
    )


# ---------------------------------------------------------------------------
# Sandhi tagging (pairwise only, following Kuang 2017)
# ---------------------------------------------------------------------------
NEUTRAL = 0


def apply_sandhi(rows: list[dict]) -> list[dict]:
    """
    For each syllable set surface_tone, sandhi_context, preceding/following tone.
    Rule: underlying T3 immediately before another underlying T3 → surface T2.
    """
    n = len(rows)
    for i, r in enumerate(rows):
        ut = r["underlying_tone"]
        prev_t = rows[i - 1]["underlying_tone"] if i > 0 else None
        next_t = rows[i + 1]["underlying_tone"] if i < n - 1 else None

        r["preceding_tone"] = prev_t
        r["following_tone"] = next_t

        if ut == 3:
            if next_t == 3:
                r["surface_tone"] = 2
                r["sandhi_context"] = "T3s"
            elif next_t == NEUTRAL:
                r["surface_tone"] = 3
                r["sandhi_context"] = "T3_neutral"
            else:
                r["surface_tone"] = 3
                r["sandhi_context"] = "T3_plain"
        else:
            r["surface_tone"] = ut
            r["sandhi_context"] = None
    return rows


# ---------------------------------------------------------------------------
# Per-turn processing
# ---------------------------------------------------------------------------
def process_turn(turn: dict, turn_idx: int, file_id: str) -> list[dict]:
    """Segment → pinyin → syllable rows for one speaker turn."""
    words = list(jieba.cut(turn["text"], cut_all=False))
    # pypinyin returns one list per character; tone style gives digit suffix
    pinyins = lazy_pinyin(turn["text"], style=Style.TONE3, neutral_tone_with_five=False)

    syl_rows = []
    char_cursor = 0
    py_cursor = 0
    for wi, word in enumerate(words):
        chars = list(word)
        py_slice = pinyins[py_cursor : py_cursor + len(chars)]
        for si, (ch, py) in enumerate(zip(chars, py_slice)):
            row = syllable_rows(ch, py, word_idx=wi, syl_idx=si)
            row.update(
                file_id=file_id,
                speaker_id=turn["speaker_id"],
                gender=turn["gender"],
                turn_start=turn["start"],
                turn_end=turn["end"],
                turn_idx=turn_idx,
            )
            syl_rows.append(row)
        py_cursor += len(chars)

    syl_rows = apply_sandhi(syl_rows)

    # Position within turn
    total = len(syl_rows)
    for pos, r in enumerate(syl_rows):
        r["pos_in_turn"] = pos
        r["turn_length_syls"] = total
    return syl_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_token_table(manifest: pd.DataFrame, corpus_root: Path) -> pd.DataFrame:
    all_rows = []
    for _, file_row in tqdm(manifest.iterrows(), total=len(manifest), desc="Stage 2"):
        file_id = file_row["file_id"]
        tx_path = corpus_root / file_row["transcript_path"]
        chosen = set(file_row["chosen_speakers"])

        turns = [t for t in parse_txt(tx_path) if t["speaker_id"] in chosen]
        for ti, turn in enumerate(turns):
            all_rows.extend(process_turn(turn, ti, file_id))

    return pd.DataFrame(all_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    corpus_root = Path(cfg["paths"]["corpus_root"])
    manifest = pd.read_parquet(args.manifest)

    df = build_token_table(manifest, corpus_root)
    print(f"\nTotal syllables:  {len(df):,}")
    print(f"T3 syllables:     {(df['underlying_tone'] == 3).sum():,}")
    print(f"T3s (sandhi):     {(df['sandhi_context'] == 'T3s').sum():,}")
    print(f"T3_plain:         {(df['sandhi_context'] == 'T3_plain').sum():,}")
    print(f"T3_neutral:       {(df['sandhi_context'] == 'T3_neutral').sum():,}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
