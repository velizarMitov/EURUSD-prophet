"""
MTL patch script — modifies existing cells and appends new Sections 19-22.
Run from the project root: python _mtl_patch.py
"""
import json

NB = 'notebooks/01_data_preparation.ipynb'
with open(NB, 'r', encoding='utf-8') as f:
    nb = json.load(f)


def gsrc(c):  return ''.join(c.get('source', []))
def ssrc(c, t): c['source'] = t.splitlines(keepends=True)


def code(t):
    return {'cell_type': 'code', 'metadata': {'language': 'python'},
            'source': t.splitlines(keepends=True), 'outputs': [], 'execution_count': None}


def md(t):
    return {'cell_type': 'markdown', 'metadata': {'language': 'markdown'},
            'source': t.splitlines(keepends=True)}


# ──────────────────────────────────────────────────────────────────────────────
# PATCH 1 — add_advanced_features: replace single binary target with dual targets
# ──────────────────────────────────────────────────────────────────────────────
for c in nb['cells']:
    s = gsrc(c)
    if 'def add_advanced_features' in s:
        old = ("    # 7. Binary target: 1 if next bar closes HIGHER than current, else 0\n"
               "    # shift(-1) causes the last row to become NaN.\n"
               "    data['target'] = (data['close'].shift(-1) > data['close']).astype(int)")
        new = ("    # 7. Dual targets for Multi-Task Learning\n"
               "    # target_return    : continuous log-return of the next bar (regression head)\n"
               "    data['target_return']    = np.log(data['close'].shift(-1) / data['close'])\n"
               "    # target_direction : binary 1=Up / 0=Down for the next bar (classification head)\n"
               "    data['target_direction'] = (data['target_return'] > 0).astype(int)\n"
               "    # Backward-compatible alias kept for Sections 5-12\n"
               "    data['target'] = data['target_direction']")
        if old in s:
            ssrc(c, s.replace(old, new))
            print('Patched: add_advanced_features')
        break

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 2 — build_advanced_features: dual targets + update FEATURE_COLS
# ──────────────────────────────────────────────────────────────────────────────
for c in nb['cells']:
    s = gsrc(c)
    if 'def build_advanced_features' in s:
        old1 = "    df['target'] = (close.shift(-1) > close).astype(int)"
        new1 = ("    # Dual targets for Multi-Task Learning\n"
                "    df['target_return']    = np.log(close.shift(-1) / close)\n"
                "    df['target_direction'] = (df['target_return'] > 0).astype(int)\n"
                "    df['target']           = df['target_direction']  # backward-compatible alias")
        old2 = "FEATURE_COLS    = [c for c in advanced_df.columns if c != 'target']"
        new2 = ("FEATURE_COLS    = [c for c in advanced_df.columns\n"
                "                   if c not in ('target', 'target_return', 'target_direction')]")
        if old1 in s:
            s = s.replace(old1, new1)
            print('Patched: build_advanced_features targets')
        if old2 in s:
            s = s.replace(old2, new2)
            print('Patched: FEATURE_COLS')
        ssrc(c, s)
        break

# ──────────────────────────────────────────────────────────────────────────────
# PATCH 3 — Section 15 GBM data split: use target_direction
# ──────────────────────────────────────────────────────────────────────────────
for c in nb['cells']:
    s = gsrc(c)
    if "y = basic_advanced_df['target']" in s and 'n_split' in s:
        old = ("y = basic_advanced_df['target']\n"
               "X = basic_advanced_df.drop('target', axis=1)")
        new = ("y = basic_advanced_df['target_direction']\n"
               "X = basic_advanced_df.drop(columns=['target', 'target_return', 'target_direction'],\n"
               "                           errors='ignore')")
        if old in s:
            ssrc(c, s.replace(old, new))
            print('Patched: GBM data split (Section 15)')
        break

# ──────────────────────────────────────────────────────────────────────────────
# NEW CELLS — Sections 19-22
# ──────────────────────────────────────────────────────────────────────────────
new_cells = []

# ── Section 19: MTL Theory ─────────────────────────────────────────────────
new_cells.append(md(r"""## Section 19 — Multi-Task Learning: Theory & Architecture

### Mathematical Foundation

**Multi-Task Learning (MTL)** trains a single shared encoder to minimise a weighted
combination of task-specific losses simultaneously, embedding a natural inductive bias
across correlated prediction objectives:

$$
\mathcal{L}_{\text{MTL}} = \lambda_1 \cdot \underbrace{\text{MSE}(\hat{y}_{ret},\, y_{ret})}_{\text{Regression head}} + \lambda_2 \cdot \underbrace{\text{BCE}(\hat{y}_{dir},\, y_{dir})}_{\text{Classification head}}
$$

where:
- $\text{MSE} = \dfrac{1}{n}\displaystyle\sum_{i=1}^n (\hat{r}_i - r_i)^2$ — penalises return magnitude errors
- $\text{BCE} = -\dfrac{1}{n}\displaystyle\sum_{i=1}^n \bigl[y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\bigr]$ — penalises directional misclassification
- $\lambda_1 = 0.3,\ \lambda_2 = 0.7$ — asymmetric weights that prioritise the tradeable direction signal over exact magnitude

### Why MTL Reduces Generalisation Error

The shared LSTM trunk must learn latent representations that explain **both** the sign and
magnitude of the next-period return. This mutual inductive bias acts as an **implicit
regulariser**, reducing the effective VC dimension of each individual head and tightening
the Bias-Variance tradeoff bound relative to two independently trained models.

### Dual Target Variables

| Variable | Type | Formula |
|----------|------|---------|
| `target_return` | Continuous regression | $r_{t+1} = \ln\!\left(P_{t+1}/P_t\right)$ |
| `target_direction` | Binary classification | $d_{t+1} = \mathbf{1}\!\left[r_{t+1} > 0\right]$ |"""))

new_cells.append(code(r"""import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

print("=== Section 19 — MTL Feature & Target Preparation ===")

# All feature columns — exclude the three target columns
MTL_FEATURE_COLS = [c for c in advanced_df.columns
                    if c not in ('target', 'target_return', 'target_direction')]

X_mtl = advanced_df[MTL_FEATURE_COLS].values
y_ret = advanced_df['target_return'].values      # continuous log-return
y_dir = advanced_df['target_direction'].values   # binary direction

print(f"Feature dimensions : {len(MTL_FEATURE_COLS)} features per bar")
print(f"Return std (full)  : {y_ret.std():.6f}")
print(f"Direction balance  : {y_dir.mean():.3f}  (proportion Up)")

# Chronological 70 / 15 / 15 split — NO shuffle (time-series rule)
n_mtl  = len(X_mtl)
tr_end = int(n_mtl * 0.70)
va_end = int(n_mtl * 0.85)

X_tr_m, X_va_m, X_te_m = X_mtl[:tr_end], X_mtl[tr_end:va_end], X_mtl[va_end:]
y_tr_ret, y_va_ret, y_te_ret = y_ret[:tr_end], y_ret[tr_end:va_end], y_ret[va_end:]
y_tr_dir, y_va_dir, y_te_dir = y_dir[:tr_end], y_dir[tr_end:va_end], y_dir[va_end:]

# Scale: fit ONLY on train to prevent leakage
scaler_mtl = StandardScaler()
X_tr_ms = scaler_mtl.fit_transform(X_tr_m)
X_va_ms = scaler_mtl.transform(X_va_m)
X_te_ms = scaler_mtl.transform(X_te_m)


def make_seqs(X, yr, yd, ts=20):
    """Build sliding-window 3-D sequences with aligned dual targets."""
    Xs, yrs, yds = [], [], []
    for i in range(len(X) - ts):
        Xs.append(X[i : i + ts])
        yrs.append(yr[i + ts - 1])
        yds.append(yd[i + ts - 1])
    return np.array(Xs), np.array(yrs), np.array(yds)


TS = 20
X_tr_seq, y_tr_seq_ret, y_tr_seq_dir = make_seqs(X_tr_ms, y_tr_ret, y_tr_dir, TS)
X_va_seq, y_va_seq_ret, y_va_seq_dir = make_seqs(X_va_ms, y_va_ret, y_va_dir, TS)
X_te_seq, y_te_seq_ret, y_te_seq_dir = make_seqs(X_te_ms, y_te_ret, y_te_dir, TS)

print(f"\nTrain sequences : {X_tr_seq.shape}")
print(f"Val   sequences : {X_va_seq.shape}")
print(f"Test  sequences : {X_te_seq.shape}")"""))

# ── Section 20: MTL LSTM ────────────────────────────────────────────────────
new_cells.append(md(r"""## Section 20 — Multi-Task LSTM via Keras Functional API

The **Keras Functional API** enables a single encoder to fork into two task-specific
output heads, each trained with its own loss function:

```
Input  (TS × F)
  └── LSTM(128, return_sequences=True) → Dropout(0.2)
  └── LSTM(64)                         → Dropout(0.2)
          ├── Dense(32, relu) → Dense(1, linear)   ← return_output
          └── Dense(32, relu) → Dense(1, sigmoid)  ← direction_output
```

Combined loss:
$$\mathcal{L}_{\text{MTL}} = 0.3 \cdot \underbrace{\text{MSE}}_{\text{return}} + 0.7 \cdot \underbrace{\text{BCE}}_{\text{direction}}$$

The higher weight on BCE ($\lambda_2 = 0.7$) reflects the practitioner priority:
**directional correctness** drives tradeable signals, while return magnitude is secondary."""))

new_cells.append(code(r"""import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

print("=== Section 20 — Multi-Task LSTM (Functional API) ===")

n_steps, n_features = X_tr_seq.shape[1], X_tr_seq.shape[2]

# ── Shared encoder ────────────────────────────────────────────────────────
inp = Input(shape=(n_steps, n_features), name='sequence_input')
x   = LSTM(128, return_sequences=True, name='lstm_shared_1')(inp)
x   = Dropout(0.2, name='dropout_1')(x)
x   = LSTM(64, name='lstm_shared_2')(x)
x   = Dropout(0.2, name='dropout_2')(x)

# ── Return head (regression) ──────────────────────────────────────────────
ret_h   = Dense(32, activation='relu', name='return_hidden')(x)
ret_out = Dense(1, activation='linear', name='return_output')(ret_h)

# ── Direction head (classification) ──────────────────────────────────────
dir_h   = Dense(32, activation='relu', name='direction_hidden')(x)
dir_out = Dense(1, activation='sigmoid', name='direction_output')(dir_h)

mtl_model = Model(inputs=inp, outputs=[ret_out, dir_out], name='MTL_LSTM')
mtl_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss={'return_output': 'mse', 'direction_output': 'binary_crossentropy'},
    loss_weights={'return_output': 0.3, 'direction_output': 0.7},
    metrics={'return_output': ['mae'], 'direction_output': ['accuracy']}
)
mtl_model.summary()

# ── Training ──────────────────────────────────────────────────────────────
es_mtl = EarlyStopping(monitor='val_loss', patience=12,
                       restore_best_weights=True, verbose=1)

hist_mtl = mtl_model.fit(
    X_tr_seq,
    {'return_output': y_tr_seq_ret, 'direction_output': y_tr_seq_dir},
    validation_data=(
        X_va_seq,
        {'return_output': y_va_seq_ret, 'direction_output': y_va_seq_dir}
    ),
    epochs=100, batch_size=64, callbacks=[es_mtl], verbose=0
)
print(f"Stopped after {len(hist_mtl.history['loss'])} epochs.")

# ── Learning curves ───────────────────────────────────────────────────────
# Dynamically resolve metric key names (vary between TF/Keras versions)
mae_k     = next((k for k in hist_mtl.history if 'return' in k and 'mae' in k and not k.startswith('val')),
                 'return_output_mae')
val_mae_k = 'val_' + mae_k
acc_k     = next((k for k in hist_mtl.history if 'direction' in k and 'acc' in k and not k.startswith('val')),
                 'direction_output_accuracy')
val_acc_k = 'val_' + acc_k

fig, axes = plt.subplots(1, 3, figsize=(18, 4))

axes[0].plot(hist_mtl.history['loss'],     color='royalblue',  label='Train')
axes[0].plot(hist_mtl.history['val_loss'], color='darkorange', label='Val')
axes[0].set_title('Combined MTL Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

if mae_k in hist_mtl.history:
    axes[1].plot(hist_mtl.history[mae_k],     color='royalblue',  label='Train MAE')
    axes[1].plot(hist_mtl.history[val_mae_k], color='darkorange', label='Val MAE')
axes[1].set_title('Return Head — MAE'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

if acc_k in hist_mtl.history:
    axes[2].plot(hist_mtl.history[acc_k],     color='royalblue',  label='Train Acc')
    axes[2].plot(hist_mtl.history[val_acc_k], color='darkorange', label='Val Acc')
axes[2].set_title('Direction Head — Accuracy'); axes[2].set_xlabel('Epoch')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle('Multi-Task LSTM Training Curves', fontsize=13)
plt.tight_layout()
plt.savefig('../results/11_mtl_lstm_curves.png', dpi=120)
plt.show()
print("Saved -> ../results/11_mtl_lstm_curves.png")"""))

# ── Section 21: Dual GBM ────────────────────────────────────────────────────
new_cells.append(md(r"""## Section 21 — Dual Gradient Boosting Pipeline

For the tree-ensemble approach we train **two independent** Gradient Boosting models,
each with a task-appropriate loss function:

| Model | Task | Loss | Rationale |
|-------|------|------|-----------|
| `GradientBoostingRegressor` | $\hat{r}_{t+1}$ (continuous) | **Huber** | Robust to extreme return outliers at market events |
| `GradientBoostingClassifier` | $\hat{d}_{t+1} \in \{0,1\}$ | **Deviance** | Calibrated directional probability estimates |

The **Huber loss** combines MSE and MAE, providing a smooth, outlier-robust objective:

$$
\mathcal{L}_{\delta}(r,\hat{r}) =
\begin{cases}
  \tfrac{1}{2}(r-\hat{r})^2 & |r-\hat{r}| \leq \delta \\
  \delta\!\left(|r-\hat{r}| - \tfrac{\delta}{2}\right) & \text{otherwise}
\end{cases}
$$"""))

new_cells.append(code(r"""from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, roc_auc_score

print("=== Section 21 — Dual GBM Pipeline ===")

# ── Task A: Regressor with Huber loss → target_return ─────────────────────
print("\n[Task A] GBM Regressor (Huber) -> target_return")
gbm_reg = GradientBoostingRegressor(
    loss='huber', n_estimators=300, max_depth=4,
    learning_rate=0.05, subsample=0.8,
    min_samples_leaf=10, random_state=RANDOM_STATE
)
gbm_reg.fit(X_tr_ms, y_tr_ret)

yhat_ret = gbm_reg.predict(X_te_ms)
mse_gbm  = mean_squared_error(y_te_ret, yhat_ret)
mae_gbm  = mean_absolute_error(y_te_ret, yhat_ret)
print(f"  Test MSE : {mse_gbm:.6e}")
print(f"  Test MAE : {mae_gbm:.6e}")

# ── Task B: Classifier → target_direction ─────────────────────────────────
print("\n[Task B] GBM Classifier -> target_direction")
gbm_clf = GradientBoostingClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, min_samples_leaf=10, random_state=RANDOM_STATE
)
gbm_clf.fit(X_tr_ms, y_tr_dir)

yhat_dir  = gbm_clf.predict(X_te_ms)
yprob_dir = gbm_clf.predict_proba(X_te_ms)[:, 1]
acc_gbm   = accuracy_score(y_te_dir, yhat_dir)
auc_gbm   = roc_auc_score(y_te_dir, yprob_dir)
print(f"  Test Accuracy : {acc_gbm:.4f}")
print(f"  Test ROC-AUC  : {auc_gbm:.4f}")"""))

# ── Section 22: Evaluation ──────────────────────────────────────────────────
new_cells.append(md(r"""## Section 22 — Multi-Task Evaluation

### Evaluation Framework

Each output head is evaluated independently with task-appropriate metrics.

**Regression head** (predicted return magnitude):
$$
\text{MSE} = \frac{1}{n}\sum_i (r_i - \hat{r}_i)^2, \qquad
\text{MAE} = \frac{1}{n}\sum_i |r_i - \hat{r}_i|
$$

**Classification head** (predicted market direction):
$$
\text{Accuracy} = \frac{TP + TN}{n}, \qquad
\text{ROC-AUC} = \int_0^1 \text{TPR}(t)\, d\text{FPR}(t)
$$"""))

new_cells.append(code(r"""from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              accuracy_score, roc_auc_score,
                              classification_report, ConfusionMatrixDisplay)
import matplotlib.pyplot as plt
import pandas as pd

print("=" * 62)
print("  MULTI-TASK EVALUATION REPORT")
print("=" * 62)

# ── MTL LSTM ──────────────────────────────────────────────────────────────
preds_mtl      = mtl_model.predict(X_te_seq, verbose=0)
yhat_ret_lstm  = preds_mtl[0].ravel()
yprob_dir_lstm = preds_mtl[1].ravel()
yhat_dir_lstm  = (yprob_dir_lstm >= 0.5).astype(int)

mse_lstm = mean_squared_error(y_te_seq_ret, yhat_ret_lstm)
mae_lstm = mean_absolute_error(y_te_seq_ret, yhat_ret_lstm)
acc_lstm = accuracy_score(y_te_seq_dir, yhat_dir_lstm)
auc_lstm = roc_auc_score(y_te_seq_dir, yprob_dir_lstm)

print("\n── MTL LSTM ─────────────────────────────────────────────")
print(f"  Return    : MSE={mse_lstm:.6e}  |  MAE={mae_lstm:.6e}")
print(f"  Direction : Accuracy={acc_lstm:.4f}  |  ROC-AUC={auc_lstm:.4f}")

# ── Dual GBM ──────────────────────────────────────────────────────────────
print("\n── Dual GBM ─────────────────────────────────────────────")
print(f"  Return    : MSE={mse_gbm:.6e}  |  MAE={mae_gbm:.6e}")
print(f"  Direction : Accuracy={acc_gbm:.4f}  |  ROC-AUC={auc_gbm:.4f}")

# ── Comparison table ──────────────────────────────────────────────────────
eval_df = pd.DataFrame({
    'MSE (Return)':   [mse_lstm,  mse_gbm],
    'MAE (Return)':   [mae_lstm,  mae_gbm],
    'Accuracy (Dir)': [acc_lstm,  acc_gbm],
    'ROC-AUC (Dir)':  [auc_lstm,  auc_gbm],
}, index=['MTL LSTM', 'Dual GBM'])

print()
display(eval_df.round(6))
eval_df.to_csv('../results/mtl_comparison_table.csv')
print("Saved -> ../results/mtl_comparison_table.csv")

# ── Confusion matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ConfusionMatrixDisplay.from_predictions(
    y_te_seq_dir, yhat_dir_lstm, display_labels=['Down', 'Up'],
    cmap='Blues', colorbar=False, ax=axes[0])
axes[0].set_title('MTL LSTM — Direction Head (Test)')

ConfusionMatrixDisplay.from_predictions(
    y_te_dir, yhat_dir, display_labels=['Down', 'Up'],
    cmap='Oranges', colorbar=False, ax=axes[1])
axes[1].set_title('Dual GBM — Direction Classifier (Test)')

plt.suptitle('Multi-Task Confusion Matrices', fontsize=13)
plt.tight_layout()
plt.savefig('../results/12_mtl_confusion.png', dpi=120)
plt.show()
print("Saved -> ../results/12_mtl_confusion.png")

print("\n── Direction classification reports ─────────────────────")
print("  [MTL LSTM]")
print(classification_report(y_te_seq_dir, yhat_dir_lstm, target_names=['Down (0)', 'Up (1)']))
print("  [Dual GBM]")
print(classification_report(y_te_dir, yhat_dir, target_names=['Down (0)', 'Up (1)']))"""))

new_cells.append(code(r"""import joblib, os

os.makedirs('../models', exist_ok=True)

# Dual GBM artifacts
joblib.dump(gbm_reg,    '../models/gbm_regressor_return.pkl')
joblib.dump(gbm_clf,    '../models/gbm_classifier_direction.pkl')
joblib.dump(scaler_mtl, '../models/scaler_mtl.pkl')

# MTL LSTM  (try .keras format first, fall back to .h5 for older TF)
try:
    mtl_model.save('../models/mtl_lstm.keras')
    print("Saved -> ../models/mtl_lstm.keras")
except Exception:
    mtl_model.save('../models/mtl_lstm.h5')
    print("Saved -> ../models/mtl_lstm.h5")

print("Saved -> ../models/gbm_regressor_return.pkl")
print("Saved -> ../models/gbm_classifier_direction.pkl")
print("Saved -> ../models/scaler_mtl.pkl")"""))

# ── Append and save ──────────────────────────────────────────────────────────
nb['cells'].extend(new_cells)
print(f"\nAppended {len(new_cells)} new cells.")

with open(NB, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook saved successfully.")
