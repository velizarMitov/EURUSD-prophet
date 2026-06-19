---
name: data-preprocessing-validation
description: 'Data Scientist role strictly adhering to statistical best practices for EURUSD time-series preprocessing. Use when: loading and cleaning OHLCV data, engineering features (technical indicators, lag features), splitting data without leakage, scaling/normalizing, checking stationarity, validating dataset integrity before model training. Triggers: data preparation, feature engineering, train test split, data leakage, missing values, normalization, stationarity, OHLCV, technical indicators, lag features, data validation.'
argument-hint: 'Describe the preprocessing task (e.g. "engineer lag features", "check for data leakage", "validate splits")'
---

# Data Preprocessing & Validation Integrity

## Role
You are a Data Scientist strictly following statistical best practices for financial time-series data. Data integrity and the **strict prohibition of data leakage** are non-negotiable.

## Non-Negotiable Rules

> **Rule #1 — NEVER evaluate the model on training data. Always hold out a test set.**
>
> Use a minimum 70% train / 30% test split. For time-series data like EURUSD, **always use chronological splitting, not random sampling**. Shuffling breaks the temporal order and introduces future leakage.

> **Rule #2 — Standardize features before regularization or neural networks.**
>
> Always apply `StandardScaler` (or equivalent) before fitting Ridge, Lasso, SVR, or any neural network. Regularization penalizes coefficients by magnitude — unscaled features make the penalty meaningless.

> **Rule #3 — Handle outliers carefully, especially for linear models.**
>
> Before fitting any linear model, inspect the residual distribution. If extreme outliers are present, consider RANSAC (RANdom Sample Consensus) to fit a robust estimator that excludes them automatically.

## When to Use
- Loading and cleaning raw EURUSD OHLCV data
- Engineering technical indicator and lag features
- Creating temporally correct train/validation/test splits
- Scaling and normalizing features
- Running stationarity and integrity checks
- Validating the feature matrix before passing it to any model

---

## The Golden Rule: No Data Leakage

> **Any information from time T+1 or later must never be used to construct features or targets at time T.**

Violations that cause silent leakage in financial ML:
- Using `df.fillna(method='ffill')` after split (fit imputer on train only)
- Fitting `StandardScaler` on the full dataset before splitting
- Rolling windows that look forward (`min_periods` set incorrectly)
- Using `train_test_split(shuffle=True)` on time-series data

---

## Step-by-Step Preprocessing Pipeline

### Step 1 — Data Acquisition

```python
import yfinance as yf
import pandas as pd

df = yf.download("EURUSD=X", start="2018-01-01", end="2024-12-31", interval="1d")
df.to_csv("data/raw/eurusd_daily.csv")
```

Alternatives: `pandas_datareader`, broker APIs, Dukascopy CSV exports.

**Validation checks after loading:**
- [ ] Index is `DatetimeIndex`, sorted ascending
- [ ] No duplicate timestamps
- [ ] Columns: Open, High, Low, Close, Volume (or Adj Close)
- [ ] Date range matches expectations

### Step 2 — Missing Value Audit

```python
print(df.isnull().sum())
print(f"Missing %: {df.isnull().mean() * 100}")
```

| Situation | Correct Action |
|-----------|---------------|
| Weekends / holidays | Expected gaps — do NOT fill |
| Isolated single NaN | Forward-fill (fit on train only) |
| Consecutive NaNs > 5 | Investigate source; consider dropping |
| Volume = 0 for FX | Normal — volume unreliable for FX pairs |

### Step 3 — Feature Engineering

All features must be computed **before** the split, then the split is applied to the full feature matrix. Never compute features referencing post-split data.

#### Lag Features (most important for time-series ML)

```python
for lag in [1, 2, 3, 5, 10, 20]:
    df[f'close_lag_{lag}'] = df['Close'].shift(lag)
```

#### Returns

```python
df['return_1d'] = df['Close'].pct_change(1)
df['return_5d'] = df['Close'].pct_change(5)
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
```

#### Technical Indicators (use `ta` or `pandas_ta`)

```python
import pandas_ta as ta

df['rsi_14']      = ta.rsi(df['Close'], length=14)
df['ema_20']      = ta.ema(df['Close'], length=20)
df['ema_50']      = ta.ema(df['Close'], length=50)
df['macd']        = ta.macd(df['Close'])['MACD_12_26_9']
df['bb_upper']    = ta.bbands(df['Close'])['BBU_5_2.0']
df['bb_lower']    = ta.bbands(df['Close'])['BBL_5_2.0']
df['atr_14']      = ta.atr(df['High'], df['Low'], df['Close'], length=14)
```

#### Target Variable Definition

| Task | Target | Code |
|------|--------|------|
| Next-day price regression | $P_{t+1}$ | `df['target'] = df['Close'].shift(-1)` |
| Next-day direction (classification) | $1$ if $P_{t+1}>P_t$ else $0$ | `df['target'] = (df['Close'].shift(-1) > df['Close']).astype(int)` |
| Next-day return regression | $r_{t+1}$ | `df['target'] = df['Close'].pct_change().shift(-1)` |

**Document your choice and justify it in the notebook.**

### Step 4 — Drop NaN Rows

```python
df.dropna(inplace=True)  # After all feature construction
print(f"Final shape: {df.shape}")
```

### Step 5 — Temporal Train / Validation / Test Split

**Never use `train_test_split` with `shuffle=True` on time-series.**

```python
n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train = df.iloc[:train_end]
val   = df.iloc[train_end:val_end]
test  = df.iloc[val_end:]

print(f"Train: {train.index[0]} → {train.index[-1]}  ({len(train)} rows)")
print(f"Val:   {val.index[0]}   → {val.index[-1]}    ({len(val)} rows)")
print(f"Test:  {test.index[0]}  → {test.index[-1]}   ({len(test)} rows)")
```

Recommended split ratios:

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Fit model |
| Validation | 15% | Tune hyperparameters |
| Test | 15% | Final evaluation ONCE |

### Step 5b — Outlier Handling (for Linear Models)

Before fitting Ridge, Lasso, or LinearRegression, check for extreme outliers:

```python
import matplotlib.pyplot as plt

# Visual inspection of returns for extreme spikes
plt.figure(figsize=(10, 3))
plt.plot(train['return_1d'])
plt.title("Daily Returns — inspect for extreme spikes")
plt.show()

# Option A: Winsorize (clip at 1st/99th percentile)
from scipy.stats import mstats
train['return_1d'] = mstats.winsorize(train['return_1d'], limits=[0.01, 0.01])

# Option B: RANSAC for robust linear fitting (excludes outliers automatically)
from sklearn.linear_model import RANSACRegressor, LinearRegression

ransac = RANSACRegressor(
    estimator=LinearRegression(),
    min_samples=0.8,           # Use at least 80% of data as inliers
    residual_threshold=None,   # Auto-set from data MAD
    random_state=42
)
ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
print(f"RANSAC: {inlier_mask.sum()} inliers / {len(inlier_mask)} total ({inlier_mask.mean()*100:.1f}%)")
```

**When to choose RANSAC:** If residual plots show a few extreme observations pulling the regression line. RANSAC ignores them by design. Document outlier treatment choice in the notebook with a justification cell.

### Step 6 — Scaling / Normalization

> **Rule #2 in practice: Always apply `StandardScaler` before regularization or neural networks.**

**Fit scaler on train only. Transform train, val, and test.**

Failure to scale before regularization (Ridge, Lasso, SVR) or neural networks causes the penalty term to treat high-magnitude features as more important than low-magnitude ones — completely invalidating the regularization effect.

```python
from sklearn.preprocessing import StandardScaler

feature_cols = [c for c in df.columns if c != 'target']

scaler = StandardScaler()
X_train = scaler.fit_transform(train[feature_cols])
X_val   = scaler.transform(val[feature_cols])
X_test  = scaler.transform(test[feature_cols])

y_train = train['target'].values
y_val   = val['target'].values
y_test  = test['target'].values

# Save scaler for inference
import joblib
joblib.dump(scaler, 'models/scaler.pkl')
```

### Step 7 — Stationarity Check (for statistical models)

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(train['Close'])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"p-value: {result[1]:.4f}")
# p < 0.05 → stationary (reject unit root)
# If non-stationary → use returns instead of price levels
```

---

## Validation Integrity Checklist

Run before passing data to any model:

```
[ ] Index sorted ascending, no duplicates
[ ] No NaN values in X_train, X_val, X_test, y_*
[ ] test.index[-1] > val.index[-1] > train.index[-1]   (temporal order)
[ ] Scaler fitted ONLY on train
[ ] Rolling/lag features use shift() not look-ahead
[ ] Target column NOT present in feature_cols
[ ] target = shift(-1) rows dropped from end
[ ] Feature shapes: X_train.shape[1] == X_val.shape[1] == X_test.shape[1]
```

```python
# Automated integrity assertions
assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], "Feature count mismatch"
assert not pd.DataFrame(X_train).isnull().any().any(), "NaN in X_train"
assert train.index.max() < val.index.min(), "Train/val temporal overlap"
assert val.index.max() < test.index.min(), "Val/test temporal overlap"
print("All integrity checks passed ✓")
```

---

## Feature Importance Sanity Check

After a first quick model, verify features make sense:

```python
import matplotlib.pyplot as plt
importances = pd.Series(model.feature_importances_, index=feature_cols)
importances.nlargest(15).plot(kind='barh')
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("results/feature_importance.png")
```

If lag features dominate → healthy for FX.
If calendar features dominate → suspect leakage.

---

## Common Statistical Mistakes

| Mistake | Consequence | Fix |
|---------|-------------|-----|
| Shuffle split on time-series | Future data in train | Use temporal split |
| Scale before split | Data leakage from test stats | Fit scaler on train only |
| Not dropping NaN after shift | NaN targets infect model | `dropna()` after all shifts |
| Using Close price as feature AND target | Trivial correlation, useless model | Use lag of Close as feature, future Close as target |
| No stationarity check | ARIMA/statistical models fail | ADF test before fitting |
