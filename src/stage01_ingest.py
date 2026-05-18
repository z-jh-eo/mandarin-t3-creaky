"""
Stage 1a — corpus ingestion (MagicData-RAMC).

Assumed layout (under corpus_root):
    SPKINFO.txt           # CHANNEL  SPEAKER_ID  GENDER  AGE  REGION,CITY  DEVICE
    UTTERANCEINFO.txt     # CHANNEL  UTTRANS_ID  SPEAKER_ID-1  SPEAKER_ID-2
                          # TOPIC  VALID(min)  TOTAL(min)  Environment
    WAV/<file_id>.wav
    TXT/<file_id>.txt     # transcript format inspected in Stage 2

Each .wav is a dyadic dialogue with TWO speakers. The manifest is
file-level (one row per dialogue) and records both speakers. Turn-level
speaker resolution happens in Stage 2 when transcripts are parsed.

Run via:
    make ingest
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
import yaml
from tqdm import tqdm


# ---------------------------------------------------------------------------
def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Speaker and utterance metadata
# ---------------------------------------------------------------------------
def load_speakers(corpus_root: Path) -> pd.DataFrame:
    """Parse SPKINFO.txt → DataFrame indexed by speaker_id."""
    df = pd.read_csv(corpus_root / "SPKINFO.txt", sep="\t", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={"region,city": "region_city"})
    return df.set_index("speaker_id")


def load_utterances(corpus_root: Path) -> pd.DataFrame:
    """Parse UTTERANCEINFO.txt → DataFrame indexed by uttrans_id (= file_id)."""
    df = pd.read_csv(corpus_root / "UTTERANCEINFO.txt", sep="\t", engine="python")
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(columns={
        "speaker_id-1": "speaker_1",
        "speaker_id-2": "speaker_2",
        "valid(min)":   "valid_minutes",
        "total(min)":   "total_minutes",
    })
    df["uttrans_id"] = df["uttrans_id"].astype(str).str.strip().str.replace(r"\.wav$", "", regex=True)
    return df.set_index("uttrans_id")


# ---------------------------------------------------------------------------
# Audio metadata
# ---------------------------------------------------------------------------
def get_audio_metadata(wav: Path) -> tuple:
    """(duration_sec, sample_rate, channels) or (None, None, None) on error."""
    try:
        info = sf.info(str(wav))
        return info.duration, info.samplerate, info.channels
    except Exception as e:
        print(f"  ! could not read {wav.name}: {e}")
        return None, None, None


# ---------------------------------------------------------------------------
# Manifest assembly
# ---------------------------------------------------------------------------
def build_manifest(config: dict) -> pd.DataFrame:
    corpus_root = Path(config["paths"]["corpus_root"]).expanduser().resolve()
    if not corpus_root.exists():
        raise FileNotFoundError(
            f"corpus_root does not exist: {corpus_root}\n"
            f"Edit `paths.corpus_root` in config.yaml."
        )

    print("Reading SPKINFO.txt and UTTERANCEINFO.txt...")
    spk_info = load_speakers(corpus_root)
    utt_info = load_utterances(corpus_root)


    print(f"  {len(spk_info)} speakers, {len(utt_info)} dialogue entries.\n")

    wav_dir = corpus_root / "WAV"
    txt_dir = corpus_root / "TXT"
    wavs = sorted(wav_dir.rglob("*.wav"))
    print(f"Found {len(wavs)} .wav files under {wav_dir.relative_to(corpus_root)}.\n")
    if not wavs:
        raise RuntimeError("No .wav files under WAV/.")

    rows = []
    n_missing_tx = n_missing_utt = n_unreadable = 0

    for wav in tqdm(wavs, desc="Building manifest"):
        file_id = wav.stem
        dur, sr, ch = get_audio_metadata(wav)
        if dur is None:
            n_unreadable += 1
            continue

        # Transcript: TXT/<file_id>.txt (or .TextGrid as a fallback)
        transcript = None
        for ext in (".txt", ".TextGrid"):
            cand = txt_dir / f"{file_id}{ext}"
            if cand.exists():
                transcript = cand
                break
        if transcript is None:
            n_missing_tx += 1

        # Speaker pair from UTTERANCEINFO
        if file_id in utt_info.index:
            u = utt_info.loc[file_id]
            s1, s2 = u["speaker_1"], u["speaker_2"]
            topic = u.get("topic")
            valid = u.get("valid_minutes")
            total = u.get("total_minutes")
            env   = u.get("environment")
        else:
            n_missing_utt += 1
            s1 = s2 = topic = valid = total = env = None

        d1 = spk_info.loc[s1].to_dict() if s1 in spk_info.index else {}
        d2 = spk_info.loc[s2].to_dict() if s2 in spk_info.index else {}

        rows.append({
            "file_id":         file_id,
            "audio_path":      str(wav.relative_to(corpus_root)),
            "transcript_path": str(transcript.relative_to(corpus_root)) if transcript else None,
            "speaker_1":       s1,
            "gender_1":        d1.get("gender"),
            "age_1":           d1.get("age"),
            "region_1":        d1.get("region_city"),
            "speaker_2":       s2,
            "gender_2":        d2.get("gender"),
            "age_2":           d2.get("age"),
            "region_2":        d2.get("region_city"),
            "topic":           topic,
            "valid_minutes":   valid,
            "total_minutes":   total,
            "environment":     env,
            "duration_sec":    dur,
            "sample_rate":     sr,
            "channels":        ch,
        })

    df = pd.DataFrame(rows)
    _print_summary(df, n_missing_tx, n_missing_utt, n_unreadable)
    return df


def _print_summary(
    df: pd.DataFrame, n_missing_tx: int, n_missing_utt: int, n_unreadable: int
) -> None:
    total_h = df["duration_sec"].sum() / 3600
    # Unique speakers across both columns
    unique_speakers = (
        pd.concat([df["speaker_1"], df["speaker_2"]]).dropna().unique()
    )
    # Gender breakdown by unique speaker (not by file)
    g_by_spkr = (
        pd.concat([
            df[["speaker_1", "gender_1"]].rename(columns={"speaker_1": "s", "gender_1": "g"}),
            df[["speaker_2", "gender_2"]].rename(columns={"speaker_2": "s", "gender_2": "g"}),
        ])
        .dropna(subset=["s"])
        .drop_duplicates("s")
    )

    print("\n" + "=" * 64)
    print("Manifest summary")
    print("=" * 64)
    print(f"  Files included:               {len(df):>6,}")
    print(f"  Audio unreadable:             {n_unreadable:>6,}  (excluded)")
    print(f"  Missing transcripts:          {n_missing_tx:>6,}")
    print(f"  Missing UTTERANCEINFO row:    {n_missing_utt:>6,}")
    print(f"  Total duration:               {total_h:>6.2f} h")
    print(f"  Unique speakers:              {len(unique_speakers):>6,}")
    if not g_by_spkr.empty:
        print(f"  Speaker gender breakdown:     {g_by_spkr['g'].value_counts().to_dict()}")
    if df["sample_rate"].notna().any():
        print(f"  Sample rates:                 {df['sample_rate'].value_counts().to_dict()}")
    print("=" * 64)


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    df = build_manifest(config)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\nWrote manifest to {out}")


if __name__ == "__main__":
    main()
