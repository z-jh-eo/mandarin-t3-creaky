"""
Stage 3a — prepare MFA input and generate run scripts.
Input:  manifest_subset.parquet + transcripts_tone.parquet
Output: data/interim/mfa_input/<speaker_id>/<file_id>.lab + symlinked WAV
        run_mfa_local.sh   (run in conda env with MFA installed)
        run_mfa_colab.ipynb (upload to Colab, run, pull TextGrids back)
Run:    python -m src.stage03a_mfa_prep --config config.yaml
"""
from __future__ import annotations
import argparse, json, os, textwrap
from pathlib import Path
import pandas as pd
import yaml


def load_config(path: str) -> dict:
    return yaml.safe_load(open(path, encoding="utf-8"))


# ---------------------------------------------------------------------------
# .lab file content: MFA expects the transcript as plain space-separated
# characters (or words). We use the turn text joined by spaces so MFA
# can align at the character level with the Mandarin dictionary.
# ---------------------------------------------------------------------------
def build_lab_content(turns: pd.DataFrame) -> str:
    """Concatenate all turn texts for one speaker in one file."""
    return " ".join(turns["char"].tolist())


# ---------------------------------------------------------------------------
# MFA input layout:
#   mfa_input/
#     <speaker_id>/
#       <file_id>.wav  -> symlink to corpus WAV
#       <file_id>.lab  -> plain-text transcript
# MFA treats each subdirectory as one speaker, which gives us
# speaker-level acoustic models and cleaner TextGrid output.
# ---------------------------------------------------------------------------
def prepare_mfa_input(manifest: pd.DataFrame, tokens: pd.DataFrame,
                      corpus_root: Path, mfa_input: Path) -> list[dict]:
    """Write lab files and WAV symlinks. Returns list of prepared entries."""
    prepared = []
    for _, row in manifest.iterrows():
        file_id = row["file_id"]
        wav_src = corpus_root / row["audio_path"]

        for spkr in row["chosen_speakers"]:
            spkr_dir = mfa_input / spkr
            spkr_dir.mkdir(parents=True, exist_ok=True)

            # Symlink WAV
            wav_link = spkr_dir / f"{file_id}.wav"
            if not wav_link.exists():
                wav_link.symlink_to(wav_src.resolve())

            # Lab file: characters for this speaker's turns only
            spkr_tokens = tokens[
                (tokens["file_id"] == file_id) &
                (tokens["speaker_id"] == spkr)
            ]
            if spkr_tokens.empty:
                continue
            lab_text = build_lab_content(spkr_tokens)
            (spkr_dir / f"{file_id}.lab").write_text(lab_text, encoding="utf-8")
            prepared.append({"file_id": file_id, "speaker_id": spkr,
                             "n_chars": len(spkr_tokens)})

    return prepared


# ---------------------------------------------------------------------------
# Shell script — local run
# ---------------------------------------------------------------------------
LOCAL_SCRIPT = """\
#!/usr/bin/env bash
# Run MFA alignment locally (inside conda env with MFA installed).
# Usage: bash run_mfa_local.sh
set -euo pipefail

MFA_INPUT="{mfa_input}"
MFA_OUTPUT="{mfa_output}"
DICTIONARY="mandarin_china_mfa"
ACOUSTIC_MODEL="mandarin_mfa"
JOBS={jobs}

echo "=== Downloading models (skipped if already present) ==="
mfa model download acoustic  $ACOUSTIC_MODEL
mfa model download dictionary $DICTIONARY

echo "=== Running alignment ==="
mfa align \\
    "$MFA_INPUT" \\
    "$DICTIONARY" \\
    "$ACOUSTIC_MODEL" \\
    "$MFA_OUTPUT" \\
    --jobs $JOBS \\
    --clean \\
    --output_format long_textgrid

echo "=== Done. TextGrids in $MFA_OUTPUT ==="
"""


# ---------------------------------------------------------------------------
# Colab notebook — minimal cells
# ---------------------------------------------------------------------------
def colab_notebook(mfa_input_rel: str, mfa_output_rel: str) -> dict:
    """Return a minimal .ipynb dict for Colab MFA run."""
    cells = [
        ("markdown", "# MFA Alignment — Colab\n"
         "1. Upload `mfa_input/` to your Google Drive under `mandarin-t3-creak/`.\n"
         "2. Run all cells.\n"
         "3. Download `mfa_output/` back to `data/interim/mfa_output/` locally."),
        ("code", """\
from google.colab import drive
drive.mount('/content/drive')
PROJECT = '/content/drive/MyDrive/mandarin-t3-creak'
MFA_INPUT  = f'{PROJECT}/mfa_input'
MFA_OUTPUT = f'{PROJECT}/mfa_output'
"""),
        ("code", """\
!conda install -y -c conda-forge montreal-forced-aligner 2>/dev/null | tail -1
!mfa model download acoustic  mandarin_mfa
!mfa model download dictionary mandarin_china_mfa
"""),
        ("code", """\
!mfa align \\
    "$MFA_INPUT" \\
    mandarin_china_mfa \\
    mandarin_mfa \\
    "$MFA_OUTPUT" \\
    --jobs 4 \\
    --clean \\
    --output_format long_textgrid
print("Done — download mfa_output/ from Drive back to data/interim/mfa_output/")
"""),
    ]

    def cell(kind, src):
        if kind == "markdown":
            return {"cell_type": "markdown", "metadata": {},
                    "source": src.splitlines(keepends=True)}
        return {"cell_type": "code", "metadata": {}, "outputs": [],
                "execution_count": None,
                "source": src.splitlines(keepends=True)}

    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                    "name": "python3"}},
        "cells": [cell(k, s) for k, s in cells],
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",    required=True)
    ap.add_argument("--manifest",  required=True)
    ap.add_argument("--tokens",    required=True)
    ap.add_argument("--mfa-input", default="data/interim/mfa_input")
    ap.add_argument("--mfa-output",default="data/interim/mfa_output")
    args = ap.parse_args()

    cfg         = load_config(args.config)
    corpus_root = Path(cfg["paths"]["corpus_root"])
    manifest    = pd.read_parquet(args.manifest)
    tokens      = pd.read_parquet(args.tokens)
    mfa_input   = Path(args.mfa_input)
    mfa_output  = Path(args.mfa_output)
    mfa_output.mkdir(parents=True, exist_ok=True)

    print("Preparing MFA input directory...")
    prepared = prepare_mfa_input(manifest, tokens, corpus_root, mfa_input)
    print(f"  {len(prepared)} speaker-file pairs prepared in {mfa_input}")

    # Local shell script
    script = LOCAL_SCRIPT.format(
        mfa_input=str(mfa_input.resolve()),
        mfa_output=str(mfa_output.resolve()),
        jobs=max(1, os.cpu_count() - 1),
    )
    Path("run_mfa_local.sh").write_text(script, encoding="utf-8")
    os.chmod("run_mfa_local.sh", 0o755)
    print("Wrote run_mfa_local.sh")

    # Colab notebook
    nb = colab_notebook(str(mfa_input), str(mfa_output))
    Path("run_mfa_colab.ipynb").write_text(
        json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Wrote run_mfa_colab.ipynb")

    print("\nNext steps:")
    print("  Local:  bash run_mfa_local.sh")
    print("  Colab:  upload mfa_input/ + run_mfa_colab.ipynb to Drive, run, pull mfa_output/ back")
    print(f"  Then:   make parse_textgrids")


if __name__ == "__main__":
    main()
