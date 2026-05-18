# Mandarin T3 Creak: Corpus Study

A reproducible pipeline for studying creaky voice in Mandarin Tone 3
across prosodic positions and speaking rates, on the MagicData-RAMC
conversational corpus.

## What this project does

For each Tone 3 syllable in a balanced subset of MagicData-RAMC, we
estimate:

- **Continuous primary outcome:** percentage of creaky frames per
  rime, detected by `creapy`.
- **Secondary outcome:** binary creak (yes/no) and creak *type* via
  unsupervised clustering on minimal acoustic measures.

The predictors of interest are prosodic position (graded, via
following-pause duration), local speaking rate, and tone-sandhi
context (T3 plain / T3+T3 / T3 + neutral). Speaker gender, F0 range,
and surrounding tones are controls.

The pipeline is split into twelve stages (see `Makefile`), each
producing a parquet artifact consumed by the next.

## Setup

### 1. Python environment

Python 3.10 or 3.11 recommended.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Montreal Forced Aligner

MFA is easiest to install via conda:

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_china_mfa
```

(Skip until needed for Stage 3.)

### 3. creapy

```bash
pip install git+https://github.com/dasl-/creapy.git
```

(Skip until needed for Stage 8. Check the latest install instructions
from the creapy repo — the package is occasionally renamed.)

### 4. Corpus

Download MagicData-RAMC and unpack into `data/raw/MagicData-RAMC/`.
Then update `paths.corpus_root` in `config.yaml` to match the actual
extracted path.

## Running the pipeline

```bash
make help               # list targets
make setup              # create directory structure
make ingest             # Stage 1a: scan corpus, build full manifest
make subset             # Stage 1b: select balanced subset (6 h test_run)
```

To switch to the larger run (12 h, useful when Colab access is
available), edit `active_subset: larger_run` in `config.yaml` and
rerun `make subset`.

## Project layout

```
.
├── Makefile            # pipeline orchestration
├── config.yaml         # all parameters
├── requirements.txt
├── data/
│   ├── raw/            # MagicData-RAMC goes here (gitignored)
│   ├── interim/        # per-stage parquet outputs (gitignored)
│   └── processed/      # final analysis master table
├── src/                # one module per stage
├── notebooks/          # modeling and visualization (stages 11–12)
├── results/            # figures and tables
└── tests/
```

## References

- Kuang (2017). *Creaky voice as a function of tonal categories and
  prosodic boundaries.* Interspeech.
- Keating, Esposito, Garellek, Khan & Kreiman (2023). *A cross-language
  acoustic space for vocalic phonation distinctions.* Language 99(2).
- Paierl et al. (2023). `creapy`: Python tool for creak detection.
