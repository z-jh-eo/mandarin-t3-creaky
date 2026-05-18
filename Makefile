# Mandarin T3 creak study — pipeline orchestration.
# Each stage produces a parquet artifact consumed by the next.
# Rerunning a stage rebuilds only what's downstream.

PYTHON  := python
CONFIG  := config.yaml

INTERIM   := data/interim
PROCESSED := data/processed

# Stage artifacts
MANIFEST_FULL   := $(INTERIM)/manifest_full.parquet
MANIFEST_SUBSET := $(INTERIM)/manifest_subset.parquet
TRANSCRIPTS     := $(INTERIM)/transcripts_tone.parquet
TEXTGRIDS       := $(INTERIM)/t3_syllables.parquet
T3_SYLLABLES    := $(INTERIM)/t3_syllables.parquet
T3_PROSODY      := $(INTERIM)/t3_prosody.parquet
T3_RATE         := $(INTERIM)/t3_rate.parquet
T3_ACOUSTIC     := $(INTERIM)/t3_acoustic.parquet
T3_CREAK        := $(INTERIM)/t3_creak.parquet
T3_CREAK_TYPE   := $(INTERIM)/t3_creak_type.parquet
MASTER          := $(PROCESSED)/analysis_master.parquet

.PHONY: all setup ingest subset clean help

# Default: full pipeline up through the master analysis table.
all: $(MASTER)

# Stage 0: directory scaffolding.
setup:
	mkdir -p $(INTERIM) $(PROCESSED) results/figures results/tables
	@echo "Directories ready."

# ---------------------------------------------------------------------------
# Stage 1a — corpus ingestion: walk corpus_root, build manifest of files.
# ---------------------------------------------------------------------------
$(MANIFEST_FULL): src/stage01_ingest.py $(CONFIG) | setup
	$(PYTHON) -m src.stage01_ingest --config $(CONFIG) --output $@

ingest: $(MANIFEST_FULL)

# ---------------------------------------------------------------------------
# Stage 1b — subset selection: gender-balanced speakers, target duration.
# ---------------------------------------------------------------------------
$(MANIFEST_SUBSET): src/stage01_subset.py $(MANIFEST_FULL) $(CONFIG)
	$(PYTHON) -m src.stage01_subset \
		--config $(CONFIG) \
		--manifest $(MANIFEST_FULL) \
		--output $@

subset: $(MANIFEST_SUBSET)

#
# Stage 2
#

$(TRANSCRIPTS): src/stage02_transcripts.py $(MANIFEST_SUBSET) $(CONFIG)
	$(PYTHON) -m src.stage02_transcripts \
		--config $(CONFIG) \
		--manifest $(MANIFEST_SUBSET) \
		--output $@

transcripts: $(TRANSCRIPTS)


$(INTERIM)/mfa_input: src/stage03a_mfa_prep.py $(MANIFEST_SUBSET) $(TRANSCRIPTS) $(CONFIG)
	$(PYTHON) -m src.stage03a_mfa_prep \
		--config $(CONFIG) \
		--manifest $(MANIFEST_SUBSET) \
		--tokens $(TRANSCRIPTS) \
		--mfa-input $(INTERIM)/mfa_input \
		--mfa-output $(INTERIM)/mfa_output

mfa_prep: $(INTERIM)/mfa_input

$(TEXTGRIDS): src/stage03b_parse_textgrids.py $(TRANSCRIPTS) $(MANIFEST_SUBSET) $(CONFIG)
	$(PYTHON) -m src.stage03b_parse_textgrids \
		--config $(CONFIG) \
		--tokens $(TRANSCRIPTS) \
		--manifest $(MANIFEST_SUBSET) \
		--mfa-output $(INTERIM)/mfa_output \
		--output $@

parse_textgrids: $(TEXTGRIDS)

$(T3_PROSODY): src/stage04_prosody.py $(TEXTGRIDS) $(CONFIG)
	$(PYTHON) -m src.stage04_prosody \
		--config $(CONFIG) \
		--input $(TEXTGRIDS) \
		--output $@

prosody: $(T3_PROSODY)

$(T3_ACOUSTIC): src/stage05_acoustic.py $(T3_PROSODY) $(CONFIG)
	$(PYTHON) -m src.stage05_acoustic \
		--config $(CONFIG) \
		--input $(T3_PROSODY) \
		--output $@

acoustic: $(T3_ACOUSTIC)

# ---------------------------------------------------------------------------
# Stages 2–10 — placeholders, to be filled in subsequent iterations.
# Listed here so the dependency graph is visible from the start.
# ---------------------------------------------------------------------------
# $(TRANSCRIPTS):    src/stage02_transcripts.py    + $(MANIFEST_SUBSET)
# $(TEXTGRIDS):      src/stage03_align.py          + $(TRANSCRIPTS)
# $(T3_SYLLABLES):   src/stage04_extract_t3.py     + $(TEXTGRIDS)
# $(T3_PROSODY):     src/stage05_prosody.py        + $(T3_SYLLABLES)
# $(T3_RATE):        src/stage06_rate.py           + $(T3_SYLLABLES)
# $(T3_ACOUSTIC):    src/stage07_acoustic.py       + $(T3_SYLLABLES)
# $(T3_CREAK):       src/stage08_creapy.py         + $(T3_SYLLABLES)
# $(T3_CREAK_TYPE):  src/stage09_creak_types.py    + $(T3_ACOUSTIC) + $(T3_CREAK)
# $(MASTER):         src/stage10_assemble.py       + all of the above

# ---------------------------------------------------------------------------
clean:
	rm -rf $(INTERIM)/* $(PROCESSED)/*
	@echo "Interim and processed data cleared (raw kept)."

help:
	@echo "Targets:"
	@echo "  setup     Create directory structure"
	@echo "  ingest    Stage 1a — scan corpus, build full manifest"
	@echo "  subset    Stage 1b — select balanced subset"
	@echo "  all       Full pipeline (stages 1–10)"
	@echo "  clean     Remove all interim and processed data"
