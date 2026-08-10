"""Close the last gap in H_cal.1: the PAIRED bootstrap against the frozen ensemble.

    python -m src.calendar_paired_bootstrap

Requires TensorFlow (to load ``models/volatility/``). Everything else in the calendar
family is numpy-only; this script is the one place the deep-learning stack is needed, and
only to READ an already-frozen model.

--------------------------------------------------------------------------------------
WHY A PAIRED BOOTSTRAP AND NOT A DIFFERENCE OF TWO REPORTED MAEs
--------------------------------------------------------------------------------------
``results/calendar_hypothesis_log.csv`` currently records the calendar model beating the
production 5-seed ensemble by point estimates alone: 0.16209 vs 0.18594 on validation,
0.19279 vs 0.21897 on test. A difference of two separately-reported averages carries no
uncertainty, so the row reads ``PENDING-PAIRED-CI`` rather than CLEARED.

The fix is to score both models on the SAME rows, resample those rows in blocks, and
recompute the difference inside each resample — which is what the volatility family already
does (``block_len=5``, ``n=2000``).

--------------------------------------------------------------------------------------
WHICH BLOCK IS THE HONEST ONE, AND WHY IT IS THE TEST BLOCK HERE
--------------------------------------------------------------------------------------
This matters and it cuts against the more flattering number.

``models/volatility/`` was produced by ``train_production_volatility_model``, whose split is
the looser production convention: PCA and scaler fit on ``[0:80%]``, and each seed's LSTM
trains on ``[0:70%]`` **using ``[70%:80%]`` for early stopping**. So the frozen ensemble has
already touched the validation block. A paired comparison there is biased *in the ensemble's
favour*, and a calendar win on validation is therefore CONSERVATIVE, not inflated.

The registered ensemble validation figure (0.18594) came from ``volatility.run()`` — the
experiment path with the stricter inner split — and those models were never persisted, so
they cannot be re-scored row-by-row.

**Primary: the test block ``[80%:100%]``.** It is genuinely out-of-sample for the frozen
ensemble, it is the block ``vol_metrics.json`` reports on, and using it here is the one-shot
final-report use the Production Methodology rule carves out — not a search knob. Nothing is
tuned against it.

Both blocks are computed and reported; only the test block gates the verdict.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import calendar_volatility as CV

OUTDIR = Path("results/calendar")
LOG = Path("results/calendar_hypothesis_log.csv")
BLOCK_LEN = 5      # volatility family convention
N_BOOT = 2000      # volatility family convention
ALPHA = 0.05


def paired_block_bootstrap(
    y: np.ndarray,
    challenger: np.ndarray,
    baseline: np.ndarray,
    *,
    block_len: int = BLOCK_LEN,
    n_boot: int = N_BOOT,
    alpha: float = ALPHA,
    seed: int = 20260810,
) -> dict[str, float]:
    """CI for ``MAE(baseline) - MAE(challenger)``; positive favours the challenger.

    The SAME resampled row indices are used for both models inside every replicate, so the
    comparison stays paired and the common row-to-row difficulty cancels out.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    n_blocks = max(1, n // block_len)
    d = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, max(1, n - block_len), size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
        idx = idx[idx < n]
        d[i] = (
            np.abs(y[idx] - baseline[idx]).mean() - np.abs(y[idx] - challenger[idx]).mean()
        )
    lo, hi = np.nanquantile(d, [alpha / 2, 1 - alpha / 2])
    point = float(np.abs(y - baseline).mean() - np.abs(y - challenger).mean())
    return {
        "delta_mae": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "favours": "calendar" if point > 0 else "ensemble",
        "n": int(n),
    }


def main() -> None:
    from src.volatility import (  # noqa: PLC0415
        batch_predict_frozen_ensemble_vol_pct,
        build_volatility_matrix,
        load_frozen_volatility_ensemble,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("config.json").read_text(encoding="utf8"))

    # --- the family's own row set, so alignment is structural rather than hoped for ---
    feat, _X, split = build_volatility_matrix(config)
    n = split["n"]
    train_end, val_end = split["train_end"], split["val_end"]
    print(f"row set from build_volatility_matrix: n={n:,}  "
          f"{feat.index[0].date()} -> {feat.index[-1].date()}")
    print(f"  train_end(70%)={train_end}  val_end(80%)={val_end}  "
          f"n_val={val_end - train_end}  n_test={n - val_end}")

    y = feat["target_volatility_pct"].to_numpy(dtype=float)
    close = feat["close"].to_numpy(dtype=float)
    r = np.concatenate([[np.nan], np.diff(np.log(close))]) * 100.0
    r[0] = 0.0
    dow = np.asarray(feat.index.dayofweek, dtype=int)

    # --- frozen ensemble, pure inference ---
    print("\nloading frozen 5-seed ensemble (TensorFlow) ...")
    artifacts = load_frozen_volatility_ensemble()
    ens = batch_predict_frozen_ensemble_vol_pct(feat, artifacts)
    print(f"  ensemble predictions: {np.isfinite(ens).sum():,} finite of {len(ens):,} "
          f"({np.isnan(ens).sum()} warm-up NaN)")
    if len(ens) != n:
        raise RuntimeError(f"length mismatch: ensemble {len(ens)} vs feat {n}")

    # --- calendar model, fitted on [0:80%] to match the ensemble's fit boundary ---
    fit80 = np.zeros(n, dtype=bool)
    fit80[:val_end] = True
    cal_model = CV.CalendarVolatilityModel(use_dow=True).fit(r, y, dow, fit80)
    cal = cal_model.predict(r, dow)

    fit70 = np.zeros(n, dtype=bool)
    fit70[:train_end] = True
    cal70 = CV.CalendarVolatilityModel(use_dow=True).fit(r, y, dow, fit70).predict(r, dow)

    blocks = {
        "validation[70:80]": (np.arange(train_end, val_end), cal70,
                              "BIASED TOWARD THE ENSEMBLE (it early-stopped here)"),
        "test[80:100]": (np.arange(val_end, n), cal,
                         "PRIMARY — genuinely out-of-sample for the frozen ensemble"),
    }

    rows = []
    results: dict[str, dict] = {}
    for name, (idx, cal_pred, note) in blocks.items():
        ok = idx[np.isfinite(ens[idx]) & np.isfinite(cal_pred[idx]) & np.isfinite(y[idx])]
        yy, cc, ee = y[ok], cal_pred[ok], ens[ok]
        b = paired_block_bootstrap(yy, cc, ee)
        results[name] = b
        rows.append({
            "block": name, "n": len(ok),
            "calendar_mae": float(np.abs(yy - cc).mean()),
            "ensemble_mae": float(np.abs(yy - ee).mean()),
            "calendar_r2": CV.metrics(yy, cc)["r2"],
            "ensemble_r2": CV.metrics(yy, ee)["r2"],
            "delta_mae": b["delta_mae"], "ci_low": b["ci_low"], "ci_high": b["ci_high"],
            "excludes_zero": b["excludes_zero"], "favours": b["favours"], "note": note,
        })
        print(f"\n=== {name}  (n={len(ok)}) ===")
        print(f"  {note}")
        print(f"  calendar MAE {rows[-1]['calendar_mae']:.5f}   "
              f"ensemble MAE {rows[-1]['ensemble_mae']:.5f}")
        print(f"  dMAE {b['delta_mae']:+.5f}  CI95 [{b['ci_low']:+.5f}, {b['ci_high']:+.5f}]  "
              f"excludes 0: {b['excludes_zero']}  favours: {b['favours']}")

    pd.DataFrame(rows).to_csv(OUTDIR / "paired_bootstrap.csv", index=False)

    # --- registry update, gated on the TEST block only ---
    t = results["test[80:100]"]
    cleared = bool(t["excludes_zero"] and t["delta_mae"] > 0)
    verdict = (
        "CLEARED — calendar model beats the frozen 5-seed ensemble, paired CI excludes zero "
        "on the out-of-sample test block"
        if cleared else
        "DROP — paired CI does not exclude zero in the calendar model's favour on the "
        "out-of-sample test block"
    )
    if LOG.exists():
        with LOG.open(encoding="utf8") as f:
            reg = list(csv.DictReader(f))
            hdr = list(reg[0].keys())
        reg[0]["point_delta_mae"] = round(t["delta_mae"], 6)
        reg[0]["ci_dmae_low"] = round(t["ci_low"], 6)
        reg[0]["ci_dmae_high"] = round(t["ci_high"], 6)
        reg[0]["cleared_bar"] = cleared
        reg[0]["verdict"] = verdict
        v = results["validation[70:80]"]
        reg[0]["notes"] += (
            f" | PAIRED BOOTSTRAP (block_len={BLOCK_LEN}, n={N_BOOT}, volatility-family "
            f"convention), scored row-by-row against the FROZEN models/volatility/ ensemble "
            f"via pure batch inference. TEST[80:100] n={rows[1]['n']}: dMAE "
            f"{t['delta_mae']:+.5f} CI95 [{t['ci_low']:+.5f},{t['ci_high']:+.5f}]. "
            f"VALIDATION[70:80] n={rows[0]['n']}: dMAE {v['delta_mae']:+.5f} CI95 "
            f"[{v['ci_low']:+.5f},{v['ci_high']:+.5f}] — reported but NOT gating, because the "
            "frozen ensemble used that block for early stopping, so the comparison there is "
            "biased in the ensemble's favour and a calendar win on it is conservative. "
            "The verdict is decided on the test block alone."
        )
        with LOG.open("w", newline="", encoding="utf8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(reg)
        print(f"\nregistry updated -> {LOG}")

    print(f"\nVERDICT: {verdict}")
    print(f"wrote {OUTDIR / 'paired_bootstrap.csv'}")
    print(
        "\nStill outstanding after this run, and it stays in the notes: the GARCH inside the "
        "calendar model is a numpy variance-targeting fit, not the `arch` MLE the volatility "
        "family used, and the model was built and scored in one pass with no pre-registration "
        "preceding measurement."
    )


if __name__ == "__main__":
    main()
