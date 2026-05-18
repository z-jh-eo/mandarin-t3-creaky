## Setup

### 1. Python environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Montreal Forced Aligner

```bash
conda install -c conda-forge montreal-forced-aligner
mfa model download acoustic mandarin_mfa
mfa model download dictionary mandarin_china_mfa
```


### 3. creapy

```bash
pip install git+https://github.com/dasl-/creapy.git
```

### 4. Corpus

The corpus is to be downloaded from : https://www.openslr.org/123/

Download MagicData-RAMC and unpack into `data/raw/MagicData-RAMC/`.
Then update `paths.corpus_root` in `config.yaml` to match the actual
extracted path.

## Running the pipeline

```bash
make help               # list targets
make setup              # create directory structure
make ingest             # Stage 1a: scan corpus, build full manifest
make subset             # Stage 1b: select balanced subset (6 h test_run)
make transcripts        # Stage 2:  parse transcription and convert to pinyin
make mfa_prep           # Stage 3a: prepare MFA input
bash run_mfa_local.sh   # to run manually force alignment
make parse_textgrids    # Stage 3b: parse textgrids given by MFA  
make prosody            # Stage 4: assigns each T3 syll a prosodic pos
# TODO
# ...
```

To switch to the larger run,  edit `active_subset: larger_run` in `config.yaml` and
rerun `make subset`.

## (Planed) Project layout

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
