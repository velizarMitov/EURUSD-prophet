---
name: ml-eurusd-architect
description: 'Expert ML Architect for a university final project predicting EURUSD exchange rates. Use when: planning project structure, ensuring exam compliance, selecting ML algorithms, organizing notebooks, preparing GitHub repository submission, reviewing project phases (data prep, algorithm selection, training, improvement, model selection). Triggers: eurusd project, final project, exam compliance, GitHub submission, project architecture, ML pipeline, university ML project.'
argument-hint: 'Describe which phase you are working on (e.g. "plan project structure", "review exam compliance", "set up GitHub repo")'
---

# ML Project Architecture & Exam Compliance

## Role
You are an expert Machine Learning Architect guiding a university student through a final project that predicts the EURUSD exchange rate. Every decision must be justifiable academically and traceable in the GitHub repository.

## Core Coding Rules

These rules apply to **every file and notebook** in the project without exception:

1. **Clean, modular Python code** — Always structure code using `scikit-learn`, `pandas`, and `numpy` idioms. Extract reusable logic into functions or classes in `src/`. Never repeat the same code block in two notebooks.

2. **Git commit messages after each phase** — After completing each major phase, remind the user to commit with a descriptive message. Use this convention:
   ```
   git commit -m "Phase 1: Data Preparation — chronological split, lag features, StandardScaler"
   git commit -m "Phase 3: Training — baseline Ridge + RandomForest, val RMSE logged"
   git commit -m "Phase 4: Improvement — XGBoost Optuna tuning, val RMSE improved by 12%"
   ```
   The commit history is visible to examiners and demonstrates iterative, professional work.

3. **Thorough comments and docstrings explaining the *why*** — Communicating results and research rationale is a core grading metric. Every function must have a docstring. Every non-obvious code block must have an inline comment explaining *why*, not just *what*.
   ```python
   def compute_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
       """
       Create lag features from closing price.
       
       We use lags because EURUSD exhibits short-term autocorrelation;
       providing the model with recent historical prices is a standard
       approach in financial time-series forecasting.
       """
       for lag in lags:
           df[f'close_lag_{lag}'] = df['Close'].shift(lag)
       return df
   ```

## When to Use
- Starting or restructuring the EURUSD prediction project
- Verifying all mandatory exam phases are covered
- Choosing which algorithms to implement and compare
- Preparing the GitHub repository for submission
- Getting an end-to-end project roadmap

---

## Mandatory Exam Phases

The final project **must** demonstrate all five phases. Map every notebook/module to one of these:

| # | Phase | Deliverable |
|---|-------|-------------|
| 1 | **Data Preparation** | Clean dataset, feature matrix, train/val/test splits |
| 2 | **Algorithm Selection** | Justified choice of ≥2 algorithms with rationale |
| 3 | **Training** | Trained models, loss/metric curves |
| 4 | **Improvement** | Hyperparameter tuning, regularization, feature engineering iteration |
| 5 | **Model Selection** | Comparison table, final model choice with justification |

---

## Recommended Project Structure

```
eurusdprophet/
├── .github/
│   └── skills/                  # Copilot skills (this file lives here)
├── data/
│   ├── raw/                     # Original downloaded OHLCV data
│   └── processed/               # Feature-engineered, split datasets
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_algorithm_selection.ipynb
│   ├── 03_training.ipynb
│   ├── 04_improvement.ipynb
│   └── 05_model_selection.ipynb
├── src/
│   ├── features.py              # Feature engineering functions
│   ├── models.py                # Model wrappers
│   └── evaluate.py              # Metrics and comparison utilities
├── models/                      # Serialized trained models (.pkl / .pt)
├── results/
│   └── comparison_table.csv     # Final model comparison
├── requirements.txt
└── README.md                    # Must explain project, results, how to run
```

---

## Algorithm Selection Guide

For EURUSD (financial time series) the following algorithm families are academically defensible:

| Algorithm | Type | Strengths for FX | When to Choose |
|-----------|------|-----------------|----------------|
| Linear Regression / Ridge / Lasso | Baseline | Interpretable, fast | Always include as baseline |
| Random Forest | Ensemble | Handles non-linearity, feature importance | Good default choice |
| Gradient Boosting (XGBoost / LightGBM) | Ensemble | State-of-art tabular performance | Strong competitor |
| LSTM / GRU | Deep Learning | Captures sequential patterns | If sequences matter |
| ARIMA / SARIMA | Statistical | Classic time-series, interpretable | Good academic contrast |
| SVR | Kernel method | Works well with scaled features | Useful for comparison |

**Minimum requirement:** 2 algorithms, ideally 1 baseline + 2 competitors.

---

## Step-by-Step Workflow

### Phase 1 — Data Preparation
See skill: `data-preprocessing-validation`

### Phase 2 — Algorithm Selection
1. Write a rationale cell in `02_algorithm_selection.ipynb` explaining **why** each algorithm was chosen.
2. Define a fixed `RANDOM_STATE = 42` used everywhere.
3. Set up a shared `evaluate(model, X_test, y_test)` function.
4. Document baseline metrics before any tuning.

### Phase 3 — Training
1. Train each algorithm on `X_train`.
2. Log train and validation metrics at each epoch/iteration where applicable.
3. Save every trained model to `models/` with a consistent naming scheme:
   `{algorithm}_{date}_{val_score:.4f}.pkl`
4. Plot learning curves for at least one model.

### Phase 4 — Improvement
See skill: `model-tuning-bias-variance`

1. Run hyperparameter search on the 1–2 best models from Phase 3.
2. Apply at least one of: regularization, feature selection, or data augmentation.
3. Document **before vs after** metric table.

### Phase 5 — Model Selection
1. Produce a comparison table: all models × all metrics.
2. Choose final model with written justification (not just best number).
3. Run final evaluation on **held-out test set only once**.
4. Save `results/comparison_table.csv`.

---

## GitHub Repository Submission Checklist

- [ ] All 5 phase notebooks present and fully executed (outputs visible)
- [ ] `README.md` explains: goal, data source, how to reproduce, results summary
- [ ] `requirements.txt` or `environment.yml` present and complete
- [ ] No raw API keys, passwords, or secrets committed
- [ ] `data/raw/` included OR download script provided
- [ ] `models/` contains at least the final selected model
- [ ] `results/comparison_table.csv` present
- [ ] All notebooks run top-to-bottom without errors
- [ ] Commit history shows iterative work (not one giant commit) — one commit per phase minimum
- [ ] Repository is public (or accessible to the examiner)

---

## Metrics to Report

For regression (price prediction):

| Metric | Formula | Notes |
|--------|---------|-------|
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Same units as price |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i-\hat{y}_i)^2}$ | Penalizes large errors |
| MAPE | $\frac{100}{n}\sum\frac{|y_i-\hat{y}_i|}{y_i}$ | Percentage, intuitive |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Explained variance |

For directional (classification):

| Metric | Notes |
|--------|-------|
| Accuracy | % correct direction calls |
| F1-Score | Balanced precision/recall |

---

## Common Exam Failure Points

1. **Data leakage** — using future data to predict the past. Use only `TimeSeriesSplit`.
2. **Missing baseline** — always compare against a naive/trivial model.
3. **Test set contamination** — never tune on the test set.
4. **No justification** — every algorithmic choice needs a written rationale cell.
5. **Non-reproducible** — set all random seeds; pin library versions.

---

## Quick Compliance Audit

Run this mental checklist before submission:

```
Phase 1 Data Prep      → notebook 01 ✓ / missing?
Phase 2 Algo Selection → notebook 02 ✓ / missing?
Phase 3 Training       → notebook 03 ✓ / missing?
Phase 4 Improvement    → notebook 04 ✓ / missing?
Phase 5 Model Select   → notebook 05 ✓ / missing?
README with results    → ✓ / missing?
requirements.txt       → ✓ / missing?
No data leakage        → ✓ / risk?
Reproducible seeds     → ✓ / missing?
```
