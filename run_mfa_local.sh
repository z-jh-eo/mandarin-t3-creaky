#!/usr/bin/env bash
# Run MFA alignment locally (inside conda env with MFA installed).
# Usage: bash run_mfa_local.sh
set -euo pipefail

MFA_INPUT="/Users/zhangjiahua/M1b_large_data/phonology-lab/project/data/interim/mfa_input"
MFA_OUTPUT="/Users/zhangjiahua/M1b_large_data/phonology-lab/project/data/interim/mfa_output"
DICTIONARY="mandarin_china_mfa"
ACOUSTIC_MODEL="mandarin_mfa"
JOBS=15

echo "=== Downloading models (skipped if already present) ==="
mfa model download acoustic  $ACOUSTIC_MODEL
mfa model download dictionary $DICTIONARY

echo "=== Running alignment ==="
mfa align \
    "$MFA_INPUT" \
    "$DICTIONARY" \
    "$ACOUSTIC_MODEL" \
    "$MFA_OUTPUT" \
    --jobs $JOBS \
    --clean \
    --output_format long_textgrid \
    --output_alphabet pinyin

echo "=== Done. TextGrids in $MFA_OUTPUT ==="
