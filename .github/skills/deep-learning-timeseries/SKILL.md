---
name: deep-learning-timeseries
description: 'Deep Learning Specialist (based on Goodfellow "Deep Learning") for EURUSD sequence modeling. Use when: building LSTM or GRU models for time-series forecasting, preventing vanishing gradient in RNNs, applying Early Stopping and Dropout regularization in Keras/PyTorch, choosing Adam or RMSprop optimizer, tuning sequence length and batch size, diagnosing deep learning training issues. Triggers: LSTM, GRU, RNN, neural network, deep learning, vanishing gradient, Early Stopping, Dropout, Adam, RMSprop, sequence modeling, Keras, PyTorch, time-series neural network.'
argument-hint: 'Describe the deep learning task (e.g. "build an LSTM for EURUSD", "add Dropout to prevent overfitting", "choose optimizer for sequence model")'
---

# Advanced Deep Learning & Time-Series (Phase 5)

## Role
You are a Deep Learning Specialist grounded in Ian Goodfellow's *Deep Learning* (MIT Press). Every architectural and training decision is justified from first principles — mathematical motivation is required alongside code.

## When to Use
- Building RNN / LSTM / GRU models for EURUSD forecasting
- Preventing or diagnosing overfitting in deep architectures
- Selecting and configuring optimizers (Adam, RMSprop)
- Implementing Early Stopping and Dropout
- Preparing sequence data (windowing) for neural network input
- Comparing deep models against classical baselines

---

## Core Rules

> **Rule #1 — Prioritize LSTM or GRU for sequence modeling.**
>
> Plain RNNs suffer from the vanishing gradient problem: gradients shrink exponentially through time steps, making it impossible to learn long-range dependencies. LSTM and GRU solve this with gating mechanisms that maintain a cell state across time. For EURUSD forecasting, always start with LSTM or GRU before considering simpler or more complex architectures.

> **Rule #2 — Regularize systematically with Early Stopping and Dropout.**
>
> Deep architectures overfit easily on financial data (low signal-to-noise ratio). Apply both Early Stopping (halt when val_loss stops improving) and Dropout (randomly zero units during training to force ensemble-like generalization). Never skip both — applying neither is the most common source of overfitting in student DL projects.

> **Rule #3 — Use adaptive optimizers (Adam or RMSprop).**
>
> Standard SGD requires careful manual learning rate tuning and often gets stuck in saddle points on high-dimensional loss surfaces. Adam (Adaptive Moment Estimation) and RMSprop maintain per-parameter adaptive learning rates, converging faster and more reliably on sequence problems. Default to Adam; switch to RMSprop if training is unstable.

---

## Step 1 — Sequence Data Preparation (Windowing)

Neural networks require fixed-length input sequences. Convert the feature matrix into (samples, timesteps, features) tensors.

```python
import numpy as np

def create_sequences(X: np.ndarray, y: np.ndarray, seq_len: int):
    """
    Slide a window of length seq_len over the feature matrix to create
    3D input tensors required by LSTM/GRU layers.
    
    We use a sliding window because EURUSD exhibits short-to-medium term
    autocorrelation; providing the last seq_len timesteps gives the model
    temporal context without requiring the full history each time.
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])   # shape: (seq_len, n_features)
        ys.append(y[i + seq_len])      # predict the value AFTER the window
    return np.array(Xs), np.array(ys)

SEQ_LEN = 20  # ~1 month of trading days; tune this as a hyperparameter

X_train_seq, y_train_seq = create_sequences(X_train, y_train, SEQ_LEN)
X_val_seq,   y_val_seq   = create_sequences(X_val,   y_val,   SEQ_LEN)
X_test_seq,  y_test_seq  = create_sequences(X_test,  y_test,  SEQ_LEN)

print(f"X_train_seq shape: {X_train_seq.shape}")  # (samples, seq_len, features)
print(f"y_train_seq shape: {y_train_seq.shape}")
```

**Sequence length guide:**

| SEQ_LEN | Trading interpretation | Use when |
|---------|----------------------|----------|
| 5 | 1 week | Short-term momentum |
| 20 | 1 month | Medium-term patterns |
| 60 | 1 quarter | Macro trend cycles |

---

## Step 2 — LSTM Architecture (Keras)

```python
import tensorflow as tf
from tensorflow import keras

def build_lstm(
    seq_len: int,
    n_features: int,
    lstm_units: int = 64,
    dropout_rate: float = 0.2,
    recurrent_dropout: float = 0.2
) -> keras.Model:
    """
    Build a stacked LSTM model for EURUSD regression.
    
    Architecture rationale (Goodfellow Ch. 10):
    - LSTM gates (input, forget, output) allow selective memory — crucial for
      FX series where some historical patterns are relevant and others are noise.
    - Dropout on outputs prevents co-adaptation of units (ensemble effect).
    - Recurrent dropout applies independently each timestep, unlike naive dropout
      which would destroy the temporal signal.
    - A Dense output layer with linear activation produces continuous price predictions.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_features)),
        
        # First LSTM layer — return sequences for stacking
        keras.layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout
        ),
        
        # Second LSTM layer — return single vector
        keras.layers.LSTM(
            units=lstm_units // 2,
            return_sequences=False,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout
        ),
        
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(dropout_rate),  # Final dropout before output
        keras.layers.Dense(1, activation='linear')  # Regression output
    ])
    return model

model = build_lstm(SEQ_LEN, X_train_seq.shape[2])
model.summary()
```

---

## Step 3 — GRU Architecture (lighter alternative)

```python
def build_gru(seq_len: int, n_features: int, gru_units: int = 64, dropout_rate: float = 0.2):
    """
    GRU (Gated Recurrent Unit) — a simplified variant of LSTM.
    
    GRU merges the forget and input gates into a single update gate, reducing
    parameter count by ~25%. Preferred when training data is limited (<5000 samples)
    or when LSTM overfits. Performance is often comparable on short sequences.
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_len, n_features)),
        keras.layers.GRU(gru_units, return_sequences=True, dropout=dropout_rate),
        keras.layers.GRU(gru_units // 2, return_sequences=False, dropout=dropout_rate),
        keras.layers.Dense(1, activation='linear')
    ])
    return model
```

**LSTM vs GRU decision:**

| Factor | Choose LSTM | Choose GRU |
|--------|-------------|-----------|
| Dataset size | Large (>10k samples) | Small–medium |
| Training time | Slower | ~25% faster |
| Long-range deps | Stronger (separate cell state) | Adequate for most FX tasks |
| Overfitting risk | Higher | Lower |

---

## Step 4 — Optimizer Configuration (Rule #3)

```python
# Adam — default choice for most deep learning tasks
# Goodfellow §8.5: Adam combines momentum (exponential moving avg of gradients)
# with RMSprop (exponential moving avg of squared gradients), giving per-parameter
# adaptive learning rates that work well on sparse and non-stationary data.
optimizer_adam = keras.optimizers.Adam(
    learning_rate=1e-3,   # Default; reduce to 1e-4 if loss oscillates
    beta_1=0.9,            # Momentum decay — standard default
    beta_2=0.999,          # RMS decay — standard default
    clipnorm=1.0           # Gradient clipping — essential for RNNs to prevent exploding gradients
)

# RMSprop — use if Adam shows unstable loss curves
# Better suited to RNNs according to Goodfellow §8.5.2
optimizer_rmsprop = keras.optimizers.RMSprop(
    learning_rate=1e-3,
    rho=0.9,      # Decay factor for moving average of squared gradients
    clipnorm=1.0
)

model.compile(
    optimizer=optimizer_adam,
    loss='mse',
    metrics=['mae']
)
```

**Optimizer decision guide:**

| Symptom | Action |
|---------|--------|
| Loss decreases steadily | Adam is working — keep defaults |
| Loss oscillates / spikes | Reduce `learning_rate` by 10× |
| Loss oscillates AND slow | Switch to RMSprop |
| Gradients exploding (NaN loss) | Lower `clipnorm` to 0.5 |

---

## Step 5 — Regularization: Early Stopping + Dropout (Rule #2)

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    # Early Stopping: halt training when val_loss stops improving
    # patience=15 means: stop if no improvement for 15 consecutive epochs
    # restore_best_weights: roll back to the epoch with the best val_loss
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    
    # Save the best model checkpoint automatically
    ModelCheckpoint(
        filepath='models/lstm_best.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=0
    ),
    
    # Reduce LR on plateau — fine-grained learning rate annealing
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,      # Halve the learning rate
        patience=7,
        min_lr=1e-6,
        verbose=1
    )
]

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=200,          # Set high — Early Stopping will terminate before this
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)
```

**Dropout placement rules:**

| Location | Rate | Rationale |
|----------|------|-----------|
| `LSTM(dropout=)` | 0.1–0.3 | Drops input/output connections each timestep |
| `LSTM(recurrent_dropout=)` | 0.1–0.2 | Drops recurrent connections — use sparingly |
| `Dense` hidden layers | 0.2–0.5 | Standard; higher rate for wider layers |
| Final output `Dense` | **Never** | Applying dropout to the output corrupts predictions |

---

## Step 6 — Training Diagnostics

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curves
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].axvline(
    np.argmin(history.history['val_loss']),
    color='red', linestyle='--', label='Best epoch'
)
axes[0].set_title('Loss Curves (MSE)')
axes[0].set_xlabel('Epoch')
axes[0].legend()

# MAE curves
axes[1].plot(history.history['mae'],     label='Train MAE')
axes[1].plot(history.history['val_mae'], label='Val MAE')
axes[1].set_title('MAE Curves')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.tight_layout()
plt.savefig('results/lstm_training_curves.png')
plt.show()

print(f"Best epoch: {np.argmin(history.history['val_loss']) + 1}")
print(f"Best val_loss: {min(history.history['val_loss']):.6f}")
```

**Reading training curves:**

| Pattern | Diagnosis | Action |
|---------|-----------|--------|
| Val loss decreasing then rising | Overfitting caught by Early Stopping | Increase dropout or reduce units |
| Both losses high and flat | Underfitting | More units, longer sequences, more features |
| Val loss never converges | Learning rate too high | Reduce by 10× |
| Loss = NaN after first epoch | Exploding gradients | Add `clipnorm=1.0` to optimizer |
| Val loss < train loss | Train dropout too aggressive | Reduce dropout rate |

---

## Step 7 — Evaluation & Comparison with Classical Models

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# Load best saved model
best_model = keras.models.load_model('models/lstm_best.keras')

y_pred = best_model.predict(X_test_seq).flatten()

print("=== LSTM Test Set Metrics ===")
print(f"MSE:  {mean_squared_error(y_test_seq, y_pred):.6f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test_seq, y_pred)):.6f}")
print(f"MAE:  {mean_absolute_error(y_test_seq, y_pred):.6f}")
print(f"R²:   {r2_score(y_test_seq, y_pred):.4f}")

# Add to comparison_table.csv alongside classical models
import pandas as pd
new_row = pd.DataFrame([{
    'Model': 'LSTM (tuned)',
    'Val MSE':  mean_squared_error(y_val_seq, best_model.predict(X_val_seq).flatten()),
    'Val RMSE': np.sqrt(mean_squared_error(y_val_seq, best_model.predict(X_val_seq).flatten())),
    'Val MAE':  mean_absolute_error(y_val_seq, best_model.predict(X_val_seq).flatten()),
    'Val R²':   r2_score(y_val_seq, best_model.predict(X_val_seq).flatten())
}])
comparison_df = pd.read_csv('results/comparison_table.csv')
comparison_df = pd.concat([comparison_df, new_row], ignore_index=True).sort_values('Val RMSE')
comparison_df.to_csv('results/comparison_table.csv', index=False)
display(comparison_df)
```

---

## Hyperparameter Search for LSTM/GRU (Optuna)

```python
import optuna

def lstm_objective(trial):
    """Bayesian search over LSTM architecture and training hyperparameters."""
    units       = trial.suggest_int('units', 32, 256, step=32)
    dropout     = trial.suggest_float('dropout', 0.0, 0.5)
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    seq_len     = trial.suggest_categorical('seq_len', [10, 20, 40, 60])
    batch_size  = trial.suggest_categorical('batch_size', [16, 32, 64])

    X_tr, y_tr = create_sequences(X_train, y_train, seq_len)
    X_vl, y_vl = create_sequences(X_val,   y_val,   seq_len)

    m = build_lstm(seq_len, X_tr.shape[2], lstm_units=units, dropout_rate=dropout)
    m.compile(optimizer=keras.optimizers.Adam(lr, clipnorm=1.0), loss='mse')
    m.fit(X_tr, y_tr,
          validation_data=(X_vl, y_vl),
          epochs=100,
          batch_size=batch_size,
          callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
          verbose=0)
    return min(m.history.history['val_loss'])

study = optuna.create_study(direction='minimize')
study.optimize(lstm_objective, n_trials=50)
print(f"Best LSTM hyperparameters: {study.best_params}")
```

---

## Deep Learning Checklist (Exam Phase 4/5)

```
[ ] Sequences created with correct shape (samples, seq_len, features)
[ ] LSTM or GRU chosen with written justification (vanishing gradient rationale)
[ ] Dropout applied to LSTM and Dense layers (NOT output layer)
[ ] Early Stopping configured with restore_best_weights=True
[ ] Adam or RMSprop with clipnorm=1.0 (gradient clipping for RNNs)
[ ] Training curves saved to results/
[ ] Residual plots generated on test set
[ ] LSTM result added to comparison_table.csv
[ ] Justification cell explaining optimizer and regularization choices
[ ] Model saved to models/lstm_best.keras
```
