---
name: model-tuning-bias-variance
description: 'ML Evaluator focusing on Bias-Variance tradeoff for EURUSD prediction models. Use when: diagnosing underfitting or overfitting, reading learning curves, running hyperparameter search (GridSearch, RandomSearch, Bayesian), applying regularization, selecting cross-validation strategy for time-series, comparing models before final selection, improving model performance. Triggers: bias variance, overfitting, underfitting, hyperparameter tuning, learning curves, cross validation, regularization, model improvement, GridSearchCV, RandomizedSearchCV, Optuna, TimeSeriesSplit.'
argument-hint: 'Describe the tuning task (e.g. "diagnose overfitting in RandomForest", "tune XGBoost with Optuna", "compare models")'
---

# Model Tuning & Bias-Variance Tradeoff

## Role
You are an ML Evaluator whose primary lens is the Bias-Variance tradeoff. Every tuning decision is grounded in diagnosing the correct failure mode first, then applying the right remedy.

## When to Use
- Diagnosing whether a model underfits or overfits
- Choosing and interpreting learning curves
- Running hyperparameter searches on time-series data
- Applying regularization techniques
- Selecting cross-validation strategy
- Comparing candidate models for Phase 5 (Model Selection)

---

## Bias-Variance Decomposition

$$
\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}
$$

| Symptom | Diagnosis | Dominant Problem | Fixes |
|---------|-----------|-----------------|-------|
| High train error, high val error | Model too simple | **High Bias (Underfitting)** | Add features, reduce regularization, use more complex model |
| Low train error, high val error | Model too complex | **High Variance (Overfitting)** | Remove features, increase regularization, gather more data |
| Low train error, low val error | Good generalization | Balanced | Proceed to fine tuning |
| Both errors high and similar | Noisy data or wrong features | Irreducible noise | Revisit feature engineering |

---

## Step 1 — Diagnose Before You Tune

Always compute the bias-variance signal before touching hyperparameters.

```python
from sklearn.metrics import mean_squared_error
import numpy as np

train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
val_rmse   = np.sqrt(mean_squared_error(y_val,   model.predict(X_val)))

gap = val_rmse - train_rmse
print(f"Train RMSE: {train_rmse:.5f}")
print(f"Val   RMSE: {val_rmse:.5f}")
print(f"Gap:        {gap:.5f}  → {'Overfitting' if gap > 0.001 else 'Underfitting or Balanced'}")
```

**Decision tree:**

```
Is train_rmse high?
├── YES → Underfitting (High Bias)
│         Fixes:
│         • Add more features (feature engineering)
│         • Reduce regularization (lower alpha / lambda)
│         • Use a more complex model (e.g. RF instead of Linear)
│         • Increase max_depth, n_estimators, or network layers
└── NO  → Is val_rmse >> train_rmse?
          ├── YES → Overfitting (High Variance)
          │         Fixes:
          │         • Remove irrelevant features / apply feature selection
          │         • Increase regularization (higher alpha / lambda / dropout)
          │         • Gather or augment more training data
          │         • Reduce max_depth, add min_samples_leaf
          └── NO  → Model is well-calibrated → proceed to search
```

---

## Step 2 — Plot Learning Curves

Learning curves are the most informative single diagnostic plot.

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    estimator=model,
    X=X_train, y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

train_mean = -train_scores.mean(axis=1)
val_mean   = -val_scores.mean(axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, label='Train RMSE')
plt.plot(train_sizes, val_mean,   label='Validation RMSE')
plt.fill_between(train_sizes,
                 train_mean - train_scores.std(axis=1),
                 train_mean + train_scores.std(axis=1), alpha=0.1)
plt.xlabel("Training Set Size")
plt.ylabel("RMSE")
plt.title(f"Learning Curve — {type(model).__name__}")
plt.legend()
plt.tight_layout()
plt.savefig(f"results/learning_curve_{type(model).__name__}.png")
plt.show()
```

### Reading Learning Curves

| Pattern | Meaning | Action |
|---------|---------|--------|
| Large gap, both plateau | Overfitting | Regularize or simplify |
| Both curves high and converge | Underfitting | More complexity or features |
| Val curve still descending at right edge | More data would help | Collect more data |
| Both curves low and converge | Well-fitted | Proceed to fine tuning |

---

## Step 3 — Cross-Validation for Time Series

**For time-series (EURUSD): use `TimeSeriesSplit`. Never use `KFold` — it shuffles temporal order.**

**For classification sub-tasks (directional prediction): use `StratifiedKFold` to preserve class balance across folds.**

```python
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold

# Time-series regression / forecasting
tscv = TimeSeriesSplit(n_splits=5)

# Directional classification (up/down)
skf = StratifiedKFold(n_splits=5, shuffle=False)  # shuffle=False preserves time order

# Always combine with GridSearchCV to avoid leaking test knowledge
from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=tscv,                               # or skf for classification
    scoring='neg_root_mean_squared_error',  # or 'f1' for classification
    n_jobs=-1
)
gs.fit(X_train, y_train)
# Best params found without ever touching the hold-out test set
print(f"Best CV score: {-gs.best_score_:.5f}")
```

**Visualize the splits:**
```python
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"Fold {fold+1}: train={len(train_idx)}, val={len(val_idx)}")
```

---

## Step 4 — Hyperparameter Search Strategies

Choose the strategy based on the number of hyperparameters and compute budget:

| Strategy | Best For | Tool |
|----------|----------|------|
| Grid Search | ≤3 params, small ranges | `GridSearchCV` |
| Random Search | 3–6 params, wide ranges | `RandomizedSearchCV` |
| Bayesian Optimization | Many params, expensive models | `Optuna` |

### Grid Search (Small Models)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

gs = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    n_jobs=-1,
    verbose=1
)
gs.fit(X_train, y_train)
print(f"Best params: {gs.best_params_}")
print(f"Best CV RMSE: {-gs.best_score_:.5f}")
```

### Random Search (Medium Models)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_dist = {
    'n_estimators': randint(50, 500),
    'max_depth': randint(2, 15),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4)
}

rs = RandomizedSearchCV(
    estimator=XGBRegressor(random_state=42),
    param_distributions=param_dist,
    n_iter=50,
    cv=TimeSeriesSplit(n_splits=5),
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rs.fit(X_train, y_train)
```

### Bayesian Optimization with Optuna (Best for LSTM / XGBoost)

```python
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 2, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
    }
    model = XGBRegressor(**params, random_state=42)
    scores = cross_val_score(
        model, X_train, y_train,
        cv=TimeSeriesSplit(n_splits=5),
        scoring='neg_root_mean_squared_error'
    )
    return -scores.mean()

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)
print(f"Best trial: {study.best_params}")
```

---

## Step 5 — Regularization Reference

| Algorithm | Regularization Knob | Effect |
|-----------|---------------------|--------|
| Ridge | `alpha ↑` | Shrinks coefficients toward zero |
| Lasso | `alpha ↑` | Sparsity — feature selection |
| Random Forest | `max_depth ↓`, `min_samples_leaf ↑` | Reduces tree depth/complexity |
| XGBoost | `lambda`, `alpha`, `max_depth ↓`, `subsample ↓` | L2/L1 + tree constraints |
| LSTM | `Dropout(rate)`, `recurrent_dropout` | Prevents co-adaptation of units |
| SVR | `C ↓` | Wider margin, more regularization |

---

## Step 6 — Before vs After Improvement Table

**Required for exam Phase 4.** Document every tuning experiment.

```python
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    train_pred = model.predict(X_train)
    val_pred   = model.predict(X_val)
    results.append({
        'Model': name,
        'Train MSE':  mean_squared_error(y_train, train_pred),
        'Val MSE':    mean_squared_error(y_val, val_pred),
        'Val RMSE':   np.sqrt(mean_squared_error(y_val, val_pred)),
        'Val MAE':    mean_absolute_error(y_val, val_pred),
        'Val R²':     r2_score(y_val, val_pred)
    })

comparison_df = pd.DataFrame(results).sort_values('Val RMSE')
comparison_df.to_csv("results/comparison_table.csv", index=False)
display(comparison_df)
```

**Always report MSE, MAE, and R² together.** Each metric reveals a different failure mode:
- **MSE/RMSE**: sensitive to large errors (good for catching dangerous mispredictions)
- **MAE**: robust to outliers, intuitive in price units
- **R²**: proportion of variance explained (0–1, negative means worse than mean predictor)

## Step 6b — Residual Plots

**Always generate residual plots** to check for non-random structure in errors. Systematic patterns indicate the model is missing something.

```python
import matplotlib.pyplot as plt
import scipy.stats as stats

val_residuals = y_val - model.predict(X_val)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: Residuals vs Fitted
axes[0].scatter(model.predict(X_val), val_residuals, alpha=0.4, s=10)
axes[0].axhline(0, color='red', linewidth=1)
axes[0].set_xlabel("Fitted Values")
axes[0].set_ylabel("Residuals")
axes[0].set_title("Residuals vs Fitted")

# Plot 2: Residual distribution
axes[1].hist(val_residuals, bins=40, edgecolor='black')
axes[1].set_title("Residual Distribution")
axes[1].set_xlabel("Residual")

# Plot 3: Q-Q plot (normality check)
stats.probplot(val_residuals, dist='norm', plot=axes[2])
axes[2].set_title("Q-Q Plot of Residuals")

plt.suptitle(f"Residual Analysis — {type(model).__name__}")
plt.tight_layout()
plt.savefig(f"results/residuals_{type(model).__name__}.png")
plt.show()
```

**Reading residual plots:**

| Pattern | Meaning | Action |
|---------|---------|--------|
| Random scatter around 0 | Good fit | No action needed |
| Funnel shape (heteroscedasticity) | Variance increases with prediction | Log-transform target or use robust loss |
| Curve / trend | Non-linearity not captured | Add polynomial features or use non-linear model |
| Outlier cluster | Extreme events | Consider RANSAC or outlier removal |
| Q-Q tails diverge | Heavy-tailed errors | Expect larger prediction errors at extremes |

---

## Step 7 — Final Model Selection Criteria

Do not select purely on RMSE. Apply a multi-criteria decision:

| Criterion | Weight | Notes |
|-----------|--------|-------|
| Val RMSE / MAE | High | Primary performance metric |
| Train-Val gap | Medium | Proxy for generalization |
| Training time | Low | Practical constraint |
| Interpretability | Medium | Required for academic justification |
| Stability across CV folds | High | Low variance = reliable model |

**Write a justification cell in `05_model_selection.ipynb` explaining the choice.**

---

## Tuning Anti-Patterns

| Anti-Pattern | Risk | Fix |
|-------------|------|-----|
| Tune on test set | Overfitting to test | Use val set for tuning only |
| Report only best run | Cherry-picking | Show CV mean ± std |
| Tune all models to same depth | Miss domain fit | Tune each independently |
| Skip baseline comparison | No reference point | Always keep untouched baseline |
| Over-tune a weak algorithm | Diminishing returns | Know when to move to a better algorithm |
