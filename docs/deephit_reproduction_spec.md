# DeepHit Reproduction Specification

## 1. Reproduction Objective

The objective of this pipeline is to reconstruct the DeepHit injury-forecasting methodology described in:

> V. Catterall, C. Midoglu, and S. Lynch,  
> "Time-to-Injury Forecasting in Elite Female Football: A DeepHit Survival Approach,"  
> *arXiv preprint arXiv:2601.19479*, Jan. 2026.  
> doi: 10.48550/arXiv.2601.19479.

The goal is to reproduce the **methodology** as faithfully as possible before introducing modifications or improvements.

The original study used non-public objective injury reports for Team B. Because these labels are not included in the public SoccerMon dataset, this project should be considered a:

> **Methodological reproduction using publicly available SoccerMon injury labels.**

Exact numerical reproduction of the reported model performance is therefore not expected.

---

## 2. Original Study Cohort

### Confirmed from the Paper

| Property | Value |
|---|---|
| Dataset | SoccerMon |
| Team | Team B |
| Study period | 2020-2021 |
| Players | 37 |
| Injuries | 43 |
| Acute injuries | 24 |
| Overuse injuries | 19 |
| Player-date observations | 4,449 |
| Features after preparation | 39 |
| Players present from beginning | 21 of 37 |

### Reproduction Limitation

The original study used separate objective Team B injury reports rather than the publicly available self-reported injury data.

Our reproduction will therefore initially use:

```text
Team B
+
Public SoccerMon injury labels
```

This difference must be documented when comparing our results with those reported in the original paper.

---

## 3. Input Data Modalities

The original study combines subjective and objective athlete-monitoring data.

### Subjective Data

Source:

```text
PmSys Athlete Monitoring System
```

Includes wellness and training-session information.

### Objective Data

Source:

```text
STATSports APEX GNSS
```

Includes GPS and heart-rate measurements collected during training and match sessions.

---

# 4. Paper Feature Set

## 4.1 Training-Load Features

The paper includes:

```text
daily_load
ATL
weekly_load
monotony
strain
ACWR
CTL28
CTL42
```

Important definitions include:

### Daily Load

```text
daily_load = sum of session RPE values for a player-day
```

### Acute Training Load

```text
ATL = average training load over the previous 7 days
```

### Weekly Load

```text
weekly_load = sum of training load over the previous 7 days
```

### CTL28

```text
CTL28 = training load accumulated over the previous 28 days
```

### CTL42

```text
CTL42 = training load accumulated over the previous 42 days
```

The exact formulas for all workload variables should be verified against the authors' implementation before reproduction.

---

## 4.2 Subjective Features

The paper includes:

```text
fatigue
mood
readiness
sleep_duration
soreness
stress
RPE
sRPE
duration_subj
```

Session RPE is calculated as:

```text
sRPE = RPE * session_duration
```

### Important Note

The public SoccerMon dataset also contains `sleep_quality`.

However, sleep quality is not included in the paper's listed DeepHit feature set.

Therefore:

```text
sleep_quality
```

should **not initially be included in the strict reproduction pipeline** unless verification of the authors' code shows otherwise.

---

## 4.3 Objective Features

The paper includes objective session features corresponding to:

```text
duration_obj

Speed_km_h_mean
Speed_km_h_max
Speed_km_h_std

sp_lir_p
sp_lir_t
sp_lir_d

sp_mir_p
sp_mir_t
sp_mir_d

sp_hir_p
sp_hir_t
sp_hir_d

sp_spr_p
sp_spr_t
sp_spr_d

Distance
distance_per_min
```

These represent:

- Objective session duration
- Mean speed
- Maximum speed
- Speed standard deviation
- Low-intensity running
- Medium-intensity running
- High-intensity running
- Sprint running
- Total distance
- Distance per minute

For each running-intensity category, the paper may include:

```text
p = proportion
t = time
d = distance
```

---

# 5. Derived Features

The paper explicitly introduces two additional engineered features.

## 5.1 Previous Injury Count

```text
past_injury_count
```

This represents the cumulative number of injuries experienced by the player **before the current observation date**.

Conceptually:

```text
March injury
      |
April injury
      |
June objective monitoring begins
      |
July observation
      |
      +--> past_injury_count = 2
```

Only historical injuries may be included.

Future injuries must never contribute to this feature.

---

## 5.2 Seven-Day Subjective Missingness

```text
subjective_missingness_7d
```

This represents the player's recent subjective questionnaire missingness over the previous seven days.

The exact denominator and implementation must be verified from the authors' source code.

### Status

```text
[ ] Exact formula needs source-code verification.
```

---

# 6. Objective Data Quality Rules

The paper reports removing physiologically implausible or likely erroneous objective observations.

The reported thresholds include:

```text
maximum speed > 32 km/h
session duration > 200 minutes
distance > 16 km
```

The reproduction pipeline should record:

```text
records_before_filtering
records_removed_speed
records_removed_duration
records_removed_distance
records_after_filtering
```

These results should be written to an audit file.

---

# 7. Missing-Data Strategies

The paper investigates three imputation approaches.

## 7.1 Player Median Imputation

Missing values are replaced using the player's median value for the corresponding feature.

Conceptually:

```text
Player A fatigue observations:

3
4
MISSING
5
4

Player median = 4

MISSING -> 4
```

---

## 7.2 Linear Interpolation

Missing values are estimated between neighboring observations.

Example:

```text
Day 1: fatigue = 3
Day 2: fatigue = missing
Day 3: fatigue = 5
```

Linear interpolation gives approximately:

```text
Day 2: fatigue = 4
```

---

## 7.3 Bespoke Imputation

The paper develops a custom teammate-relative imputation method.

The method uses approximately the previous two weeks of information to characterize how a player typically compares with teammates and uses that relationship when estimating missing observations.

The exact mathematical implementation must be recovered from the authors' source code.

### Status

```text
[ ] Exact bespoke imputation algorithm needs source-code verification.
```

---

# 8. Reported Imputation Results

For the chronological DeepHit experiment, the paper reports approximately:

| Imputation Method | C-index |
|---|---:|
| Linear interpolation | 0.660 |
| Bespoke imputation | 0.762 |

The bespoke approach produced the strongest reported DeepHit result.

These values should be treated as **reference benchmarks**, not expected reproduction values, because our injury labels differ from those used in the original study.

---

# 9. Feature Removal After Imputation

The paper reports removing features that remained substantially incomplete after imputation.

However, the manuscript does not provide a clear numerical missingness threshold.

### Status

```text
[ ] Missingness threshold needs source-code verification.
```

---

# 10. DeepHit Architecture

The primary model is:

```text
DeepHit
```

with:

```text
MLP backbone
```

The paper does **not** use a GRU, LSTM, RNN, CNN, or TSSI architecture for the reproduced DeepHit model.

These architectures should only be considered during the later improved-model phase.

Conceptually:

```text
Engineered SoccerMon features
            |
            v
           MLP
            |
            v
         DeepHit
            |
            v
     Time-to-injury risk
```

---

# 11. Survival Target

The dataset is reformatted for survival analysis using:

```text
time_to_event
event
```

The event indicator is:

```text
1 = injury event
0 = censored observation
```

As an injury approaches, time-to-event may appear as:

```text
7
6
5
4
3
2
1
```

### Unresolved Details

The following must be verified:

```text
[ ] Exact censoring implementation
[ ] Recurrent injury handling
[ ] Reset behavior following an injury
[ ] Survival duration assigned to censored observations
[ ] Handling of overlapping injury windows
```

---

# 12. Temporal Configuration

The reported DeepHit configuration uses:

```text
History / input window = 21 days
Prediction horizon     = 7 days
```

Conceptually:

```text
PAST                                      FUTURE

t-20 ------------------------- t | t+1 -------- t+7
|<--------- 21 days ---------->| |<--- 7 days --->|
             INPUT                    FORECAST
```

### Unresolved Detail

Because the model uses an MLP rather than a recurrent architecture, the exact representation of the 21-day history must be verified.

Possible implementations include:

```text
flattened 21-day feature matrix

rolling summary features

single player-day containing historical features
```

### Status

```text
[ ] Verify exact 21-day input representation.
```

---

# 13. Feature Standardization

The paper reports using:

```text
StandardScaler
```

for feature standardization.

For leakage-safe reproduction:

```text
TRAIN
  |
  +--> fit StandardScaler
  |
  v
transform training data

TEST
  |
  +--> use training scaler
```

The test dataset must not be used to calculate the training mean or standard deviation.

### Status

```text
[ ] Verify authors' exact scaling implementation.
```

---

# 14. Chronological Validation

The primary experiment uses an:

```text
80% training
20% testing
```

chronological split.

Conceptually:

```text
TIME ---------------------------------------------------->

|------------------ 80% ----------------|----- 20% ------|

                 TRAIN                         TEST
```

The observations must not be randomly shuffled before this split.

### Unresolved Detail

Determine whether the authors performed:

```text
global chronological split
```

or:

```text
per-player chronological split
```

### Status

```text
[ ] Verify chronological split implementation.
```

---

# 15. Leave-One-Player-Out Validation

The paper also performs:

```text
Leave-One-Player-Out (LOPO)
```

For each fold:

```text
All players except Player A
            |
            v
          TRAIN

Player A
   |
   v
  TEST
```

The process is repeated for each eligible player.

The purpose is to test whether the model generalizes to an athlete it has never seen during training.

Reported findings include approximately:

```text
C-index IQR = 0.192

Maximum individual-player C-index = 0.974

Correlation between C-index and sessions:
r = 0.44

Correlation between C-index and injury count:
r = -0.08
```

---

# 16. Evaluation Metric

The primary DeepHit evaluation metric is:

```text
Concordance Index
```

or:

```text
C-index
```

The C-index evaluates whether the survival model correctly ranks observations according to relative event timing/risk.

Important:

```text
C-index = 0.762
```

does **not** mean:

```text
76.2% injury classification accuracy
```

It indicates approximately 76.2% concordance among comparable survival pairs.

---

# 17. Explainability

The paper uses:

```text
SHAP
```

to investigate model predictions.

The reproduction should eventually examine:

```text
global feature importance

player-specific feature importance

individual high-risk dates
```

SHAP should only be implemented after the DeepHit reproduction itself has been validated.

---

# 18. Primary Paper Benchmark

The main reported configuration is approximately:

```text
Model:
DeepHit

Backbone:
MLP

History:
21 days

Prediction horizon:
7 days

Scaling:
StandardScaler

Imputation:
Bespoke

Validation:
80/20 chronological split

Metric:
C-index

Reported C-index:
0.762
```

This is the primary methodological reference point for the reproduction.

---

# 19. Parameters Requiring Source-Code Verification

Before model implementation, the following parameters should be recovered from the authors' public code where possible.

## Neural Network

```text
[ ] Number of hidden layers
[ ] Hidden-layer sizes
[ ] Activation functions
[ ] Dropout
[ ] Batch normalization, if any
```

## Training

```text
[ ] Optimizer
[ ] Learning rate
[ ] Epochs
[ ] Batch size
[ ] Early stopping
[ ] Random seed
```

## DeepHit

```text
[ ] Alpha
[ ] Sigma
[ ] Duration discretization
[ ] Number of duration bins
[ ] Loss configuration
```

## Imputation

```text
[ ] Exact bespoke formula
[ ] Exact linear interpolation behavior
[ ] Exact player-median behavior
[ ] Edge handling for missing values
```

## Missingness

```text
[ ] Exact subjective_missingness_7d formula
[ ] Post-imputation feature-removal threshold
```

## Survival Dataset

```text
[ ] Exact time-to-event construction
[ ] Exact censoring logic
[ ] Recurrent injury handling
[ ] Post-injury handling
```

## Temporal Representation

```text
[ ] Exact 21-day input representation
[ ] Exact 7-day prediction-horizon construction
```

## Validation

```text
[ ] Global vs per-player chronological split
[ ] Exact LOPO implementation
[ ] StandardScaler fitting sequence
```

---

# 20. Known Reproduction Limitation

The largest unavoidable difference is:

```text
Original Paper
      |
      +--> Team B objective injury reports
              |
              +--> not publicly available
```

versus:

```text
Our Reproduction
      |
      +--> public SoccerMon injury labels
```

Therefore, model-performance differences cannot automatically be interpreted as failures to reproduce the model.

The primary goal is to reproduce the **methodology** faithfully.

---

# 21. Phase 1 Completion Checklist

Phase 1 is complete when:

```text
[x] Paper model identified
[x] Paper cohort documented
[x] Feature families documented
[x] Objective QA rules documented
[x] Imputation approaches documented
[x] Temporal windows documented
[x] Validation approaches documented
[x] Evaluation metric documented
[x] Known dataset limitation documented

[x] Authors' public repository inspected
    - Repository exists at:
      https://github.com/simulamet-host/soccermon-deephit
    - As inspected in September 2026, the repository contains only:
      - README.md
      - LICENSE
    - No model implementation, preprocessing scripts, notebooks,
      configuration files, or hyperparameter definitions are publicly
      available in the repository.

[!] The following parameters could therefore not be recovered
    from the public repository and must remain documented as
    reconstruction choices unless another authoritative source
    becomes available.
[ ] Missing hyperparameters recovered
[ ] Bespoke imputation verified
[ ] Survival target construction verified
[ ] Chronological split verified
[ ] 21-day representation verified
```

---

# 22. Next Phase

After completing the remaining source-code verification tasks:

```text
Phase 2:
Construct the DeepHit reproduction cohort
```

The reproduction pipeline will then begin transforming the audited SoccerMon data into the player-day dataset required by DeepHit.
