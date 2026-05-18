"""
Stage 1b — balanced subset selection (speaker-centric).

Each MagicData-RAMC dialogue has two speakers, so we pick speakers
first, then collect files where the chosen speakers appear. For each
chosen speaker we add files (in random order) until the cumulative
per-speaker speech time meets `minutes_per_speaker`.

Per-speaker speech time per file is approximated as half the file
duration (the dialogue is roughly balanced between the two speakers).
Stage 2 will refine this with actual turn timings.

Output columns: same as Stage 1a's manifest, plus
    chosen_speakers:  list of chosen speakers present in this file
                      (one or both — Stage 2 keeps only their turns).

Run via:
    make subset
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_gender(value) -> str | None:
    """Map any gender label to 'M' / 'F' / None."""
    if pd.isna(value):
        return None
    s = str(value).strip().upper()
    if s in {"M", "MALE", "1", "男"}:
        return "M"
    if s in {"F", "FEMALE", "0", "2", "女"}:
        return "F"
    return None


def _speaker_to_file_index(manifest: pd.DataFrame) -> pd.DataFrame:
    """Long-format table: one row per (file, speaker) pair.

    Columns: file_id, speaker, gender, duration_sec.
    """
    parts = []
    for spkr_col, gender_col in [("speaker_1", "gender_1"), ("speaker_2", "gender_2")]:
        sub = manifest[["file_id", spkr_col, gender_col, "duration_sec"]].copy()
        sub.columns = ["file_id", "speaker", "gender", "duration_sec"]
        parts.append(sub)
    long = pd.concat(parts, ignore_index=True).dropna(subset=["speaker"])
    long["gender"] = long["gender"].map(_normalize_gender)
    return long


def _pick_balanced_speakers(
    long: pd.DataFrame, n_speakers: int, rng: np.random.Generator
) -> np.ndarray:
    """Choose n speakers, 50/50 gender split when possible."""
    by_speaker = (
        long.dropna(subset=["gender"])
        .drop_duplicates("speaker")
        .set_index("speaker")["gender"]
    )

    n_per = n_speakers // 2
    males = by_speaker[by_speaker == "M"].index.to_numpy()
    females = by_speaker[by_speaker == "F"].index.to_numpy()

    if len(males) >= n_per and len(females) >= n_per:
        return np.concatenate([
            rng.choice(males, n_per, replace=False),
            rng.choice(females, n_per, replace=False),
        ])

    print(
        f"WARNING: not enough gendered speakers "
        f"(need {n_per}M/{n_per}F, have {len(males)}M/{len(females)}F). "
        f"Falling back to uniform sampling."
    )
    all_spkrs = by_speaker.index.to_numpy()
    if len(all_spkrs) < n_speakers:
        raise ValueError(
            f"Only {len(all_spkrs)} speakers with known gender available; need {n_speakers}."
        )
    return rng.choice(all_spkrs, n_speakers, replace=False)


# ---------------------------------------------------------------------------
# Subset selection
# ---------------------------------------------------------------------------
def select_subset(manifest: pd.DataFrame, config: dict) -> pd.DataFrame:
    spec = config["subsets"][config["active_subset"]]
    n_speakers = spec["speakers"]
    rng = np.random.default_rng(config["random_seed"])

    long = _speaker_to_file_index(manifest)
    chosen = _pick_balanced_speakers(long, n_speakers, rng)
    print(f"\nChose {len(chosen)} speakers.")

    # MagicData-RAMC: each speaker appears in exactly one dialogue file,
    # so for each chosen speaker we simply take all files where they appear
    # (in practice: exactly one file). The greedy duration cap is intentionally
    # absent — if the corpus ever has multi-file speakers, all their files
    # are included, keeping things simple and reproducible.
    selected_files: set[str] = set()
    speaker_files: dict[str, set[str]] = {}   # file_id -> {chosen speakers in this file}

    for spkr in chosen:
        pool = long[long["speaker"] == spkr]
        if pool.empty:
            print(f"  ! speaker {spkr} has no files; skipping")
            continue
        for _, row in pool.iterrows():
            selected_files.add(row["file_id"])
            speaker_files.setdefault(row["file_id"], set()).add(spkr)

    subset = manifest[manifest["file_id"].isin(selected_files)].copy()
    # Store chosen_speakers as a sorted list (parquet-safe via pyarrow)
    subset["chosen_speakers"] = subset["file_id"].map(
        lambda fid: sorted(speaker_files.get(fid, set()))
    )
    _print_summary(subset, chosen)
    return subset


def _print_summary(subset: pd.DataFrame, chosen: np.ndarray) -> None:
    total_h = subset["duration_sec"].sum() / 3600

    # Files per chosen speaker
    files_per_speaker = Counter()
    for sp_list in subset["chosen_speakers"]:
        for sp in sp_list:
            files_per_speaker[sp] += 1
    counts = np.array(list(files_per_speaker.values())) if files_per_speaker else np.array([0])

    # Estimated per-speaker speech time (≈ half of each shared file)
    speech_per_speaker = {}
    durs = dict(zip(subset["file_id"], subset["duration_sec"]))
    for fid, sp_list in zip(subset["file_id"], subset["chosen_speakers"]):
        for sp in sp_list:
            speech_per_speaker[sp] = speech_per_speaker.get(sp, 0.0) + durs[fid] / 2.0
    speech_vals = np.array(list(speech_per_speaker.values())) / 60 if speech_per_speaker else np.array([0.0])

    print("\n" + "=" * 64)
    print(f"Subset summary ({len(chosen)} chosen speakers)")
    print("=" * 64)
    print(f"  Files in subset:              {len(subset):>6,}")
    print(f"  Total file duration:          {total_h:>6.2f} h")
    print(f"  Est. speech / speaker (min):  "
          f"min={speech_vals.min():.1f}, median={np.median(speech_vals):.1f}, "
          f"max={speech_vals.max():.1f}")
    print(f"  Files per chosen speaker:     "
          f"min={counts.min()}, median={int(np.median(counts))}, max={counts.max()}")
    print("=" * 64)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_parquet(args.manifest)
    subset = select_subset(manifest, config)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(out, index=False)
    print(f"\nWrote subset manifest to {out}")


if __name__ == "__main__":
    main()
