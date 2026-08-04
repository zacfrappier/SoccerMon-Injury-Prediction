This repository contains the code for developing machine learning and deep learning models to predict sports injuries using the SoccerMon dataset.

The project investigates both traditional machine learning methods and modern survival analysis approaches, with an emphasis on reproducible preprocessing, feature engineering, and model interpretability.

## Objectives

- Build a reproducible injury prediction pipeline
- Compare classical ML models with deep learning approaches
- Evaluate survival analysis models (e.g., DeepHit)
- Investigate feature importance using SHAP
- Reproduce and extend recent SoccerMon research

## Repository Structure

```text
SoccerMon-Injury-Prediction/

├── data/          # Raw and processed datasets (ignored by Git)
├── notebooks/     # Exploratory analyses
├── src/           # Source code
├── models/        # Saved trained models
├── results/       # Figures and evaluation outputs
├── docs/          # Papers and project documentation
└── README.md
```

## Dataset

The SoccerMon dataset is not included in this repository due to licensing and size restrictions.

Expected directory:

```text
data/
    raw/
    processed/
```

See the project documentation for instructions on obtaining the dataset.

## Installation

Clone the repository

```bash
git clone <repository_url>
cd SoccerMon-Injury-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

## Current Status

Project initialization

- [x] Repository created
- [x] Import SoccerMon dataset
- [ ] Exploratory data analysis
- [ ] Feature engineering
- [ ] Baseline models
- [ ] DeepHit implementation
- [ ] SHAP analysis

## Roadmap

Phase 1
- Dataset exploration
- Data cleaning
- Missing value analysis

Phase 2
- Feature engineering
- Baseline machine learning models

Phase 3
- Deep learning models
- Survival analysis

Phase 4
- Explainability
- Model comparison

Phase 5
- Manuscript preparation

## License

See LICENSE.