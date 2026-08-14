# SoccerMon Injury Prediction — Audit Guide

This file documents the audit scripts under `src/audit/`, including what each script reads, what it writes, how to run it, how it works, and the main findings discovered so far.

The audit layer is intentionally separate from preprocessing and modeling:

```text
data/raw/  -->  src/audit/*.py  -->  data/processed/audit/
```

Raw data should remain unchanged. Audit scripts inspect, summarize, and validate the dataset so later preprocessing and modeling decisions are reproducible.

## General usage

Run from the repository root:

```bash
cd ~/Repos/SoccerMon-Injury-Prediction
python3.10 src/audit/<script>.py
```

---

# Objective Data Audits

## Script 1: `01_dataset_tree.py`

**Purpose:** Inventory the raw objective-data hierarchy without reading all sensor rows.

**Run:**
```bash
python3.10 src/audit/01_dataset_tree.py
```

**Inspect:**
```bash
cat data/processed/audit/objective_dataset_summary.txt
head -n 10 data/processed/audit/objective_file_manifest.csv
```

**How the code works:** Recursively scans objective folders, parses team/date/player information from filenames, records paths and file sizes, and creates one manifest row per Parquet file.

**Findings:**
1. 10,075 objective Parquet files.
2. About 160.97 GB of objective Parquet data.
3. 2 teams and 2 years: 2020 and 2021.
4. 407 unique dates.
5. 75 unique objective players.
6. No unmatched Parquet filename patterns.
7. Manifest has 10,076 lines including the header.

**Why it matters:** Later objective audits can use the manifest instead of rescanning the entire raw hierarchy.

---

## Script 2: `02_player_inventory.py`

**Purpose:** Measure objective-data coverage per player.

**Run:**
```bash
python3.10 src/audit/02_player_inventory.py
```

**Inspect:**
```bash
cat data/processed/audit/objective_player_summary.txt
head -n 10 data/processed/audit/objective_player_inventory.csv
```

**How the code works:** Groups the Script 1 manifest by team/player and calculates file count, date span, years, months, and storage.

**Findings:**
1. 75 objective players.
2. TeamA: 38 players.
3. TeamB: 37 players.
4. No player appears in both teams.
5. Session-file count ranges from 1 to 365.
6. Mean session-file count is about 134.33.
7. Objective coverage differs substantially by athlete.

**Why it matters:** Helps identify athletes with enough longitudinal sensor coverage for modeling.

---

## Script 3: `03_schema_summary.py`

**Purpose:** Verify whether objective Parquet files share a consistent schema.

**Run:**
```bash
python3.10 src/audit/03_schema_summary.py
```

**Inspect:**
```bash
cat data/processed/audit/objective_schema_summary.txt
cat data/processed/audit/objective_schema_definitions.txt
head -n 10 data/processed/audit/objective_schema_file_audit.csv
```

**How the code works:** Samples representative Parquet files with PyArrow, reads metadata/schema, and compares schema signatures.

**Findings:**
1. 100 representative files inspected.
2. 100/100 successfully read.
3. One unique schema.
4. 17 columns per file.
5. Row counts roughly 171,070 to 1,248,170.
6. Mean row count about 626,053.
7. One row group per inspected file.
8. Columns include `player_name`, `time`, GPS, speed, heart rate, accelerometer, and gyroscope fields.

**Why it matters:** Supports using one common objective preprocessing pipeline across teams/seasons.

---

## Script 4: objective value/readability audit

**Purpose:** Read actual objective sensor values and inspect data quality beyond schema metadata.

**Run:** Use the exact Script 4 filename currently present in `src/audit/`.

**How the code works:** Reads representative Parquet content and checks value readability, missingness, and suspicious values.

**Findings:**
1. Sampled objective files were readable.
2. Heart rate can contain zero values.
3. Some files contain very high fractions of zero heart-rate values.
4. Zero heart rate should likely be treated as a missing/unavailable-sensor code, pending documentation confirmation.

**Why it matters:** Prevents invalid physiological interpretation of sensor placeholder values.

---

## Script 5: sampling-frequency audit

**Purpose:** Understand objective timestamps, repeated timestamps, and gaps.

**Run:** Use the exact Script 5 filename currently present in `src/audit/`.

**How the code works:** Reads contiguous row blocks and calculates time differences between neighboring records.

**Findings:**
1. Timestamp resolution is about 0.1 seconds.
2. Roughly 10 unique timestamps occur per second.
3. Many rows share each timestamp.
4. Occasional short gaps occur.
5. Timestamp frequency is not the same as row frequency.

**Why it matters:** Avoids incorrect duration or sampling calculations.

---

## Script 6: `06_timestamp_multiplicity.py`

**Purpose:** Determine why timestamps repeat and which sensors change within a repeated timestamp.

**Run:**
```bash
python3.10 src/audit/06_timestamp_multiplicity.py
```

**Inspect:**
```bash
cat data/processed/audit/objective_timestamp_multiplicity_summary.txt
head -n 10 data/processed/audit/objective_timestamp_multiplicity_file_audit.csv
head -n 25 data/processed/audit/objective_timestamp_column_variation.csv
```

**How the code works:** Groups contiguous rows by parsed timestamp, counts rows per timestamp, estimates timestamp/row frequency, and measures variation within each timestamp group.

**Findings:**
1. About 10 rows per timestamp.
2. About 10 unique timestamps per second.
3. Effective row density is about 100 rows/second.
4. Latitude, longitude, speed, and heart rate stay constant within a timestamp group.
5. Accelerometer and gyroscope values vary inside the group.
6. Objective files appear to combine slower GPS/heart-rate measurements with higher-frequency IMU data.

**Why it matters:** GPS and heart-rate repetitions should not be treated as independent measurements, while IMU rows remain meaningful.

---

# Subjective Data Audits

## Script 7: `07_subjective_inventory.py`

**Purpose:** Inventory the subjective-data tables, schemas, rows, fields, players, and missingness.

**Run:**
```bash
python3.10 src/audit/07_subjective_inventory.py
```

**Inspect:**
```bash
cat data/processed/audit/subjective_dataset_summary.txt
head -n 25 data/processed/audit/subjective_file_inventory.csv
head -n 40 data/processed/audit/subjective_column_inventory.csv
```

**How the code works:** Recursively finds CSV files and records schema, dtypes, row counts, possible player/date columns, missingness, and duplicates.

**Findings:**
1. 18 CSV files plus `session.json`.
2. Five logical categories: game-performance, illness, injury, training-load, wellness.
3. Wellness and training-load tables are wide format.
4. Wide tables contain 50 player columns.
5. Wellness has substantial missingness.
6. Training-load tables contain no standard NaN values in the audit.
7. Injury, illness, and game-performance are event-style long tables.

**Why it matters:** Wide tables will eventually need to be reshaped into player-day form.

---

## Script 8: `08_subjective_structure.py`

**Purpose:** Validate the common structure, calendar, player sets, ranges, and missingness of subjective wide tables.

**Run:**
```bash
python3.10 src/audit/08_subjective_structure.py
```

**Inspect:**
```bash
cat data/processed/audit/subjective_structure_summary.txt
head -n 20 data/processed/audit/subjective_structure_tables.csv
head -n 20 data/processed/audit/subjective_player_missingness.csv
cat data/processed/audit/subjective_injury_duplicates.csv
```

**How the code works:** Treats the first column as date-like, identifies team-prefixed player columns, compares player/date sets, calculates missingness and value ranges, and checks injury duplicates.

**Important date rule:** SoccerMon subjective dates are `DD.MM.YYYY`; use:
```python
pd.to_datetime(values, format="%d.%m.%Y", errors="coerce")
```

**Findings:**
1. The wide tables consistently use 50 players.
2. Automatic date parsing was initially wrong; explicit `DD.MM.YYYY` parsing is required.
3. Wellness player-value missingness is around 53.5% in several files.
4. Approximate observed wellness ranges:
   - fatigue 1–5
   - mood 1–5
   - readiness 1–10
   - sleep duration 1–12
   - sleep quality 1–5
   - soreness 1–5
   - stress 1–5
5. Training-load tables contain supplied derived features including daily load, weekly load, ATL, CTL28, CTL42, ACWR, monotony, and strain.

**Why it matters:** Defines the structure that later subjective preprocessing must normalize.

---

## Script 9: `09_session_json_audit.py`

**Purpose:** Understand `training-load/session.json`.

**Run:**
```bash
python3.10 src/audit/09_session_json_audit.py
```

**Inspect:**
```bash
cat data/processed/audit/session_json_summary.txt
cat data/processed/audit/session_json_numeric_summary.csv
head -n 20 data/processed/audit/session_json_normalized_preview.csv
```

**How the code works:** Treats top-level JSON keys as player IDs and expands each player's list of sessions into one row per session.

**Findings:**
1. 16,265 session records.
2. 50 players.
3. Columns: `player_name`, `srpe`, `rpe`, `duration`, `date`.
4. No missing cells in those normalized fields.
5. 76 exact duplicate session rows.
6. RPE range 0–10.
7. Duration range 0–190.
8. sRPE range 0–1800.
9. Multiple sessions can occur on one player-day.

**Why it matters:** Provides the lower-level source needed for independently reconstructing workload metrics.

---

# Workload Validation Audits

## Script 10: `10_training_load_validation.py`

**Purpose:** Validate the sRPE formula and reconstruct daily load from `session.json`.

**Run:**
```bash
python3.10 src/audit/10_training_load_validation.py
```

**Inspect:**
```bash
cat data/processed/audit/training_load_validation_summary.txt
head -n 20 data/processed/audit/daily_load_validation.csv
```

**How the code works:** Verifies `sRPE = RPE × duration`, aggregates sRPE by player/date, and compares raw and deduplicated reconstructions against `daily_load.csv`.

**Findings:**
1. `sRPE = RPE × duration` for 16,265/16,265 sessions.
2. Formula mismatches: 0.
3. After fixing date parsing, reconstructed daily load agrees extremely closely with the provided table.
4. Removing exact session duplicates does not explain remaining discrepancies.

**Why it matters:** Establishes a reproducible workload pipeline while preserving the supplied-feature baseline.

---

## Script 11: `11_daily_load_alignment.py`

**Purpose:** Test whether remaining daily-load differences come from calendar misalignment.

**Run:**
```bash
python3.10 src/audit/11_daily_load_alignment.py
```

**Inspect:**
```bash
cat data/processed/audit/daily_load_alignment_summary.txt
cat data/processed/audit/daily_load_alignment_summary.csv
```

**How the code works:** Compares reconstructed load at same date, ±1/±2 days, and nearest available provided dates.

**Findings:**
1. Same-date alignment is correct.
2. 14,087/14,197 player-days match exactly.
3. Exact match rate: 99.23%.
4. Median absolute difference: 0.
5. Shifted dates perform around 3% or lower.
6. The provided daily-load table is not simply date-shifted.

**Why it matters:** Confirms same-calendar-day aggregation.

---

## Script 12: `12_daily_load_mismatch_categories.py`

**Purpose:** Categorize the 110 remaining mismatches.

**Run:**
```bash
python3.10 src/audit/12_daily_load_mismatch_categories.py
```

**Inspect:**
```bash
cat data/processed/audit/daily_load_mismatch_summary.txt
head -n 30 data/processed/audit/daily_load_mismatch_classification.csv
head -n 30 data/processed/audit/daily_load_unexplained_mismatches.csv
```

**How the code works:** Tests duplicate removal, zero-value cases, single-session equality, one-session differences, and small rounding differences.

**Findings:**
1. 110 mismatched player-days.
2. 109 remain unexplained by simple categories.
3. 1 differs by exactly one session amount.
4. 0 are explained by duplicate removal.

**Why it matters:** Shows the discrepancy is localized rather than a failure of the general formula.

---

## Script 13: `13_daily_load_mismatch_patterns.py`

**Purpose:** Look for player-specific and temporal patterns in the remaining 110 differences.

**Run:**
```bash
python3.10 src/audit/13_daily_load_mismatch_patterns.py
```

**Inspect:**
```bash
cat data/processed/audit/daily_load_mismatch_pattern_summary.txt
cat data/processed/audit/daily_load_mismatch_difference_frequency.csv
head -n 40 data/processed/audit/daily_load_mismatch_context_windows.csv
```

**How the code works:** Groups mismatches by player, year, weekday, difference amount, and local date windows.

**Findings:**
1. All 110 mismatches occur in only three TeamA players.
2. Distribution:
   - `TeamA-4051...`: 63
   - `TeamA-5cd7...`: 31
   - `TeamA-32fed...`: 16
3. In every mismatch, supplied daily load is greater than reconstructed daily load.
4. Strong weekday/temporal clustering exists.
5. Repeated differences often look like plausible sRPE session loads.
6. A reasonable hypothesis is that supplied daily load may include session information not present in released `session.json`; this is not proven.

**Why it matters:** Supports keeping two separate modeling pipelines instead of forcing reconstructed values to equal supplied values.

---

# Player Population Audit

## Script 14: `14_player_overlap_audit.py`

**Purpose:** Determine which players exist in objective data, subjective tables, `session.json`, and `injury.csv`, including source-specific players.

**Run:**
```bash
python3.10 src/audit/14_player_overlap_audit.py
```

**Inspect:**
```bash
cat data/processed/audit/player_overlap_summary.txt
head -n 30 data/processed/audit/player_overlap_audit.csv
```

**How the code works:** Reconstructs objective IDs as `TeamA-<uuid>`/`TeamB-<uuid>`, reads player IDs from the other sources, then creates one membership row per unique player.

**Core player definition:**
```text
objective
AND subjective wide tables
AND session.json
```

**Findings:**
1. 78 unique player IDs appear across all sources.
2. Objective data: 75 players.
3. Subjective wide tables: 50 players.
4. `session.json`: 50 players.
5. Core overlap: **47 players**.
6. 28 objective-only players.
7. 3 subjective/session-only players with no objective data.
8. All 15 injured players are present in objective data.
9. All 15 injured players are present in subjective data.
10. All 15 injured players are part of the 47-player core population.
11. Two daily-load anomaly players also have injury records.

**Why it matters:** 47 is the natural starting population for a combined objective + subjective model. The 28 objective-only players may later support an objective-only model.

---

# Injury Audits

## Script 15: `15_injury_event_audit.py`

**Purpose:** Understand the raw injury records before defining a prediction target.

**Run:**
```bash
python3.10 src/audit/15_injury_event_audit.py
```

**Inspect:**
```bash
cat data/processed/audit/injury_event_summary.txt
cat data/processed/audit/injury_type_summary.csv
head -n 30 data/processed/audit/injury_player_summary.csv
head -n 30 data/processed/audit/injury_same_day_records.csv
head -n 30 data/processed/audit/injury_objective_alignment.csv
```

**How the code works:** Parses injury records, counts duplicates/types/dates, measures recurrence spacing, and compares injury dates against each player's objective monitoring period.

**Findings:**
1. 162 raw injury records.
2. 6 duplicate rows beyond the first copy.
3. 156 unique rows after exact deduplication.
4. 15 players have injury records.
5. 108 unique injury dates.
6. Injury date range: 2020-01-14 to 2021-11-03.
7. TeamA: 151 injury rows; TeamB: 11.
8. 5 player-days contain multiple injury rows.
9. Median gap between consecutive injury records is only 2 days.
10. All 162 injury rows belong to core players.
11. Only 57/162 injury rows occur inside the corresponding player's objective-data period.
12. Only 26/162 have objective recording on the exact injury date.
13. Many injury rows occur before objective data collection begins.
14. Pre-objective injuries should be retained for later injury-history features.
15. Raw injury rows should not automatically be treated as independent injury onsets.

**Why it matters:** The target requires episode/onset interpretation, not a direct row-to-label conversion.

---

## Script 16: `16_injury_episode_audit.py`

**Purpose:** Explore how repeated injury status observations could be grouped into candidate episodes.

**Run:**
```bash
python3.10 src/audit/16_injury_episode_audit.py
```

**Inspect:**
```bash
cat data/processed/audit/injury_episode_summary.txt
cat data/processed/audit/injury_episode_gap_sensitivity.csv
head -n 30 data/processed/audit/injury_candidate_episodes_7d.csv
```

**How the code works:** Explodes multi-region injury JSON into player/date/body-region/severity observations, removes exact component duplicates, and groups repeated region/severity observations using several gap thresholds.

**Findings:**
1. 162 raw rows expand to 306 region/severity observations.
2. 299 unique player/date/region/severity observations remain.
3. Episode count is highly sensitive to gap definition:
   - 3 days: 153
   - 7 days: 108
   - 14 days: 79
   - 28 days: 69
4. Exploratory 7-day rule yields 108 candidate episodes.
5. 7-day view: 85 minor and 23 major candidate episodes.
6. Some candidate conditions span many weeks; one right-knee major episode spans 82 observed days.
7. One player alone produces 55 of the 108 7-day candidate episodes.
8. A simple fixed gap should not yet be treated as the final injury target definition.
9. Severity/body-region changes may represent evolving injury status rather than new injuries.

**Why it matters:** Injury target construction is highly sensitive to the episode rule and should be grounded in SoccerMon documentation/literature before modeling.

---

# Current Modeling Design

## Pipeline A — Provided workload variables

Use the SoccerMon-supplied variables directly:
```text
daily_load
weekly_load
ATL
CTL28
CTL42
ACWR
monotony
strain
```

## Pipeline B — Reconstructed workload variables

Build independently from released lower-level session data:
```text
RPE × duration
      ↓
     sRPE
      ↓
 daily load
      ↓
rolling workload
      ↓
ATL / CTL / ACWR / monotony / strain
```

Keep both pipelines separate, for example:
```text
data/processed/provided_features/
data/processed/reconstructed_features/
```

This enables a direct comparison between models based on the supplied SoccerMon features and models based on independently reproducible feature engineering.

---

# Important Rules Established by the Audits

1. **Subjective dates must be parsed as `DD.MM.YYYY`.**
2. **Do not modify `data/raw/`.**
3. **Do not automatically delete duplicate session rows.**
4. **Treat objective heart-rate zero values cautiously; likely missing/unavailable readings.**
5. **Do not treat every injury row as a new injury onset.**
6. **Preserve pre-objective injury information for historical predictors.**
7. **At prediction date `t`, only use information known before `t` to avoid leakage.**
8. **Keep provided and reconstructed workload pipelines separate.**

---

# Quick Audit Index

| Script | Question answered |
|---|---|
| 01 | What objective files exist? |
| 02 | Which players have objective data and how much? |
| 03 | Are Parquet schemas consistent? |
| 04 | Are representative objective values readable/plausible? |
| 05 | What is the timestamp/sampling behavior? |
| 06 | Why are timestamps repeated and which sensors vary? |
| 07 | What subjective tables exist? |
| 08 | Are the subjective wide tables aligned and what are their ranges/missingness? |
| 09 | What is inside `session.json`? |
| 10 | Can sRPE and daily load be reconstructed? |
| 11 | Are daily-load differences caused by date alignment? |
| 12 | What simple categories explain daily-load mismatches? |
| 13 | Are those mismatches player/time specific? |
| 14 | Which player IDs overlap across sources? |
| 15 | What do raw injury records represent structurally? |
| 16 | How sensitive are candidate injury episodes to grouping rules? |

---

# Template for Future Audit Documentation

```markdown
## Script N: `NN_script_name.py`

**Purpose:** What question does this audit answer?

**Primary input:** What files does it read?

**Run:**
```bash
python3.10 src/audit/NN_script_name.py
```

**Inspect:**
```bash
cat data/processed/audit/output_summary.txt
```

**Outputs:** What files does it create?

**How the code works:** Brief explanation of the important logic.

**Findings:**
1. Finding.
2. Finding.
3. Finding.

**Why it matters:** How does this affect preprocessing, feature engineering, target construction, or modeling?
```
