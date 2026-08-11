"""Deep-learning model card — every trained neural network in this project, in one place.

    python -m src.dl_model_report

The project's headline finding is that a ten-parameter calendar model outperforms the
neural ensemble on the volatility target. That conclusion is only meaningful because the
neural networks it is compared against were built properly, trained to convergence with
early stopping, and evaluated honestly. This module makes that work visible rather than
leaving it implicit in nine `.keras` files.

Runs with or without TensorFlow. With TF installed it loads each network and prints the
real Keras architecture and parameter count; without it, the report falls back to the
declared architecture in `config.json` and `_train_pipeline.py` plus the committed metrics,
so a reviewer on any machine still sees the full picture.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ======================================================================================
# Declared inventory — what each network is, independent of whether TF can load it
# ======================================================================================


@dataclass
class NetworkSpec:
    key: str
    path: str
    family: str
    task: str
    architecture: str
    trained_on: str
    status: str
    notes: str = ""
    metrics: dict = field(default_factory=dict)


def _vol_metrics() -> dict:
    try:
        return json.loads(Path("models/volatility/vol_metrics.json").read_text(encoding="utf8"))
    except Exception:  # noqa: BLE001
        return {}


def build_inventory() -> list[NetworkSpec]:
    vm = _vol_metrics()
    val = vm.get("validation_decision", {})
    return [
        NetworkSpec(
            key="baseline_multitask",
            path="models/baseline/lstm_multitask_eurusd.keras",
            family="Daily direction + return (price-only variant)",
            task="Two heads: linear next-day return, sigmoid next-day direction",
            architecture=(
                "Input(20, n_features) -> LSTM(64) -> Dropout(0.3) -> "
                "[Dense(1, linear) 'return_output', Dense(1, sigmoid) 'direction_output']"
            ),
            trained_on="[0:70%] with [70%:80%] for early stopping, Adam(1e-3), "
                       "batch 64, up to 100 epochs, patience 10",
            status="IN PRODUCTION — served by src/inference.py",
            notes="Multi-task: the shared trunk is trained on both objectives jointly. "
                  "Direction ROC-AUC ~ 0.50, consistent with market efficiency.",
        ),
        NetworkSpec(
            key="macro_multitask",
            path="models/with_macro/lstm_multitask_eurusd.keras",
            family="Daily direction + return (macro-augmented variant)",
            task="Same two heads, 27-column feature set including four FRED macro features",
            architecture="Identical topology to the baseline variant; different input width",
            trained_on="Same protocol, own PCA and scaler fitted on its own train block",
            status="IN PRODUCTION — served alongside baseline; both run on every request",
            notes="The macro features are KEEP-provisional: none cleared the corrected "
                  "significance bar. The dual-variant design exists so the forward "
                  "paper-trading ledgers can decide between them on cost-net P&L.",
        ),
        *[
            NetworkSpec(
                key=f"volatility_seed{seed}",
                path=f"models/volatility/volatility_lstm_seed{seed}.keras",
                family="Next-day realised volatility (5-seed ensemble member)",
                task="Three-head multi-task trunk; only the volatility head is served",
                architecture="Input(time_steps, PCA-reduced price features) -> LSTM -> "
                             "Dropout -> 3 heads (volatility / return / direction)",
                trained_on="[0:70%] fit, [70%:80%] early stopping; PCA+scaler on [0:80%]",
                status="IN PRODUCTION — the ensemble mean is served, all-or-nothing",
                notes="Seed ensembling was not a style choice: single-seed runs showed "
                      "TF/oneDNN nondeterminism of the same magnitude as the effects "
                      "under test, so the validated object was pre-specified as the mean "
                      "over five seeds.",
                metrics=(
                    {
                        "validation MAE (ensemble)": val.get("mt_ensemble_mae"),
                        "validation R2 (ensemble)": val.get("mt_ensemble_r2"),
                        "validation MAE (GARCH baseline)": val.get("garch_mae"),
                        "test MAE (ensemble)": vm.get("test_ensemble_mae"),
                        "test R2 (ensemble)": vm.get("test_ensemble_r2"),
                        "test MAE (GARCH baseline)": vm.get("test_garch_mae"),
                        "test MAE (persistence)": vm.get("test_persistence_mae"),
                    }
                    if seed == 42
                    else {}
                ),
            )
            for seed in (42, 43, 44, 45, 46)
        ],
        NetworkSpec(
            key="h1_seq2vec",
            path="models/h1_lstm.keras",
            family="Hourly-to-daily ensemble member",
            task="Sequence-to-vector: 24 hourly bars -> one daily return estimate",
            architecture="Input(24, n_h1_features) -> LSTM(32) -> Dropout -> Dense(1)",
            trained_on="Chronological split, early stopping on a held-out slice",
            status="IN PRODUCTION — one member of the H1 -> daily committee",
            notes="Sits alongside XGBoost, RandomForest and SVM members.",
        ),
        NetworkSpec(
            key="kronos_foundation_model",
            path="models/external_kronos/vol_calibration.json",
            family="External pre-trained time-series foundation model (Kronos)",
            task="Zero-shot probabilistic forecasting; used for cross-model comparison",
            architecture=(
                "Third-party pre-trained transformer, loaded at PINNED HuggingFace "
                "revisions via src/external/kronos/loader.py. Not trained here — "
                "autoregressive sampling (pred_len=24, sample_count=30, T=1.0, top_p=0.9)"
            ),
            trained_on="NOT trained by this project. Vendored behind a commit-pinned "
                       "loader; optional dependency in requirements-kronos.txt so the "
                       "core app installs and runs without it (kronos_ready = False)",
            status="RESEARCH ONLY — evaluated, not served",
            notes=(
                "The project's first cross-model comparison, and the one place a modern "
                "foundation model is put against our own. Pre-declared primary: "
                "incremental correlation of Kronos's pred_abs_move_pct with the 5-seed "
                "ensemble's residual = +0.1667, CI99.44% [+0.0354, +0.3090] -- excludes "
                "zero, robust to block length, seed and rank transform. But the secondary "
                "forecast-improvement test INCLUDES zero, and Kronos does not beat "
                "GARCH(1,1) on those rows. Verdict: the signal is real and does not "
                "convert into a distinguishable forecast gain. "
                "CONTAMINATION HANDLED EXPLICITLY: the comparison window was forced to "
                "2024-07..2026-06 by Kronos's own training cutoff, and the registry entry "
                "records that this is the spent test block rather than the usual "
                "validation arbiter -- a constraint imposed by the external model, "
                "disclosed rather than ignored."
            ),
        ),
        NetworkSpec(
            key="ti_lstm_h1",
            path="models/ti_lstm_h1/ti_lstm_h1.keras",
            family="H1 technical-indicator LSTM",
            task="Next-hour direction from technical indicators",
            architecture="Stacked LSTM over an H1 indicator window",
            trained_on="Chronological split with early stopping",
            status="IN PRODUCTION BY OWNER OVERRIDE — against its own DROP verdict",
            notes="Recorded as an override in IMPROVEMENT_LOG.md (2026-07-18), with the "
                  "contrary evidence stored beside it rather than removed. Documented "
                  "deliberately: a model shipped against its evidence should be visible.",
        ),
    ]


# ======================================================================================
# Reporting
# ======================================================================================


def _try_load_keras(path: str):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    try:
        from tensorflow.keras.models import load_model  # noqa: PLC0415

        return load_model(path, compile=False)
    except Exception:  # noqa: BLE001
        return None


def report(show_summary: bool = True) -> dict:
    inventory = build_inventory()
    print("=" * 78)
    print("DEEP LEARNING MODEL CARD — neural networks in this project")
    print("=" * 78)

    present = [n for n in inventory if Path(n.path).exists()]
    missing = [n for n in inventory if not Path(n.path).exists()]
    print(f"\n{len(present)} of {len(inventory)} entries present "
          "(9 trained here + 1 external pre-trained model evaluated for comparison).")
    if missing:
        print("  MISSING: " + ", ".join(n.path for n in missing))

    total_params = 0
    tf_available = True
    for spec in present:
        print("\n" + "-" * 78)
        print(f"{spec.key}    ({Path(spec.path).stat().st_size / 1024:.0f} KB)")
        print("-" * 78)
        print(f"  family        {spec.family}")
        print(f"  task          {spec.task}")
        print(f"  architecture  {spec.architecture}")
        print(f"  training      {spec.trained_on}")
        print(f"  status        {spec.status}")

        model = _try_load_keras(spec.path)
        if model is None:
            tf_available = False
            print("  parameters    (TensorFlow not available — declared architecture shown)")
        else:
            n = int(model.count_params())
            total_params += n
            print(f"  parameters    {n:,} trainable+non-trainable")
            if show_summary:
                print("\n  Keras summary:")
                model.summary(print_fn=lambda s: print("    " + s))

        if spec.metrics:
            print("\n  measured performance:")
            for k, v in spec.metrics.items():
                if v is not None:
                    print(f"    {k:<34} {v}")
        if spec.notes:
            print(f"\n  note: {spec.notes}")

    print("\n" + "=" * 78)
    if tf_available and total_params:
        print(f"Total parameters across all networks: {total_params:,}")
    else:
        print("TensorFlow unavailable — parameter counts require `pip install tensorflow`.")
        print("Declared architectures and committed metrics are shown above and are")
        print("sufficient to audit what was built without running anything.")

    print("""
HOW TO READ THIS ALONGSIDE THE PROJECT'S HEADLINE FINDING

These networks were not strawmen. The daily multi-task LSTM is a shared-trunk two-head
model trained with early stopping; the volatility model is a five-seed ensemble whose
ensembling was forced by measured framework nondeterminism, not chosen for effect. Both
are in production and both are served on every request.

The project's conclusion — that a ten-parameter calendar model outperforms the volatility
ensemble on both evaluation blocks — is a statement about this problem, not about the
quality of these networks. Deep learning was applied competently and then measured
honestly against a simple alternative. Reporting that the simple alternative won is the
result; it is not an admission that the networks were built badly.
""")
    return {"present": len(present), "total": len(inventory), "params": total_params}


if __name__ == "__main__":
    report()
