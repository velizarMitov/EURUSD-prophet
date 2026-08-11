"""Run this first. It reports what works on THIS machine and what does not.

    python verify_installation.py

No arguments, no configuration, no network required. It never trains anything and never
writes to `models/`. Every check degrades gracefully: a missing optional dependency is
reported as unavailable rather than raising, so the script always completes and always
tells you the truth about the environment it is running in.

The project is designed so the core result — the calendar volatility model — needs only
numpy and pandas. TensorFlow is required to load the neural ensemble, and MetaTrader 5 is
Windows-only and optional (the data chain falls back to yfinance and then to the bundled
CSV, both committed).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

GREEN, RED, DIM, RESET = "\033[32m", "\033[31m", "\033[2m", "\033[0m"
if sys.platform == "win32":
    GREEN = RED = DIM = RESET = ""


def ok(msg: str) -> None:
    print(f"  {GREEN}PASS{RESET}  {msg}")


def bad(msg: str) -> None:
    print(f"  {RED}FAIL{RESET}  {msg}")


def note(msg: str) -> None:
    print(f"  {DIM}....{RESET}  {msg}")


def header(msg: str) -> None:
    print(f"\n{'=' * 74}\n{msg}\n{'=' * 74}")


def main() -> int:
    failures = 0

    header("1. Environment")
    print(f"  Python {sys.version.split()[0]} on {sys.platform}")
    core = {"numpy": True, "pandas": True, "sklearn": True, "matplotlib": True}
    optional = {
        "tensorflow": "needed to load the 5-seed neural ensemble",
        "MetaTrader5": "Windows-only live feed; falls back to bundled CSV",
        "jax": "research architectures only (Idea 3)",
        "arch": "reference GARCH MLE; the calendar model has its own numpy fit",
    }
    for mod, required in core.items():
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}")
        except ImportError:
            bad(f"{mod} MISSING — required")
            failures += 1
    for mod, why in optional.items():
        try:
            m = importlib.import_module(mod)
            ok(f"{mod} {getattr(m, '__version__', '')}  ({why})")
        except Exception:
            note(f"{mod} not installed — {why}")

    header("2. Committed data and artifacts")
    checks = [
        ("results/eurusd_features.csv", "daily feature matrix"),
        ("results/eurusd_h1.csv", "hourly bars"),
        ("models/baseline/best_gbm_eurusd.pkl", "baseline GBM classifier"),
        ("models/with_macro/best_gbm_eurusd.pkl", "macro-variant GBM classifier"),
        ("models/volatility/vol_metrics.json", "volatility ensemble metrics"),
        ("models/calendar/calendar_volatility.json", "calendar model (10 parameters)"),
    ]
    for path, what in checks:
        if Path(path).exists():
            ok(f"{path}  — {what}")
        else:
            bad(f"{path} MISSING — {what}")
            failures += 1
    seeds = list(Path("models/volatility").glob("volatility_lstm_seed*.keras"))
    (ok if len(seeds) == 5 else bad)(f"volatility ensemble: {len(seeds)}/5 seed models")
    failures += len(seeds) != 5

    header("3. The headline model — reproduced live, numpy only")
    try:
        from src import calendar_volatility as CV

        data = CV.build_daily_dataset()
        n = len(data)
        tr, va, te = CV.chronological_masks(n)
        ok(f"dataset {n:,} rows  {data.index[0].date()} -> {data.index[-1].date()}")
        ok(f"splits: train {tr.sum():,} | validation {va.sum():,} | test {te.sum():,}")

        import numpy as np

        r = data["log_return_pct"].to_numpy()
        y = data["target_volatility_pct"].to_numpy()
        dow = data["dow"].to_numpy()
        model = CV.CalendarVolatilityModel(use_dow=True).fit(r, y, dow, tr)
        pred = model.predict(r, dow)
        mv = CV.metrics(y[va], pred[va])
        mt = CV.metrics(y[te], pred[te])

        print()
        print(f"  {'model':<34}{'val MAE':>10}{'val R2':>9}{'test MAE':>10}{'test R2':>9}")
        print(f"  {'calendar (this run)':<34}{mv['mae']:>10.5f}{mv['r2']:>9.4f}"
              f"{mt['mae']:>10.5f}{mt['r2']:>9.4f}")
        print(f"  {'5-seed LSTM ensemble (recorded)':<34}{0.18594:>10.5f}{0.1444:>9.4f}"
              f"{0.21897:>10.5f}{0.1098:>9.4f}")
        print(f"  {'GARCH(1,1) (recorded)':<34}{0.20379:>10.5f}{0.0094:>9.4f}"
              f"{0.23257:>10.5f}{0.0356:>9.4f}")
        print()
        # Two deliberate differences from notebooks/00_final_report.ipynb §7.1, so the two
        # documents can be compared without wondering why they disagree in the 4th decimal:
        #   * here ONE model is fitted on [0:70%] and scored on both blocks -- a fast
        #     smoke check. The report refits on [0:80%] before scoring the test block,
        #     which is the correct protocol and gives a slightly better test MAE.
        #   * the two 'recorded' rows are the historical logged figures. The report scores
        #     the CURRENT frozen models/volatility/ artifacts instead, which is why its
        #     ensemble column differs.
        print("  note: one fit on [0:70%] scored on both blocks (fast check). The report")
        print("        refits on [0:80%] for the test block and re-scores the frozen")
        print("        ensemble rather than quoting these recorded figures, so its table")
        print("        differs slightly by design — see 00_final_report.ipynb §7.1.")
        print()
        names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        p = model.params
        print(f"  the entire model: GARCH a={p.alpha:.4f} b={p.beta:.4f} scale={p.scale:.4f}")
        print("  " + "  ".join(
            f"{names[int(k)]} {v:.3f}" for k, v in sorted(p.dow_factors.items())
        ))
        if mv["mae"] < 0.18594 and mt["mae"] < 0.21897:
            ok("calendar model beats the recorded neural ensemble on BOTH blocks")
        else:
            bad("calendar model did not reproduce — investigate before believing the report")
            failures += 1
    except Exception as exc:  # noqa: BLE001
        bad(f"calendar model failed: {type(exc).__name__}: {exc}")
        failures += 1

    header("4. Production service (needs TensorFlow)")
    try:
        import tensorflow  # noqa: F401
        tf_present = True
    except Exception:  # noqa: BLE001
        tf_present = False

    if not tf_present:
        note("TensorFlow not installed — the serving path cannot be checked here")
        note("this does NOT affect section 3; the calendar model needs no deep-learning stack")
    else:
        try:
            from src.inference import PredictionService

            # PredictionService takes (base_dir, config). Calling it with no arguments
            # raised TypeError, which the old handler reported as "TensorFlow required" —
            # so a fully working install looked like a missing dependency. Report what
            # actually happened instead of guessing a cause.
            repo = Path(__file__).resolve().parent
            cfg = json.loads((repo / "config.json").read_text(encoding="utf-8"))
            svc = PredictionService(str(repo), cfg)
            for flag in ("baseline_ready", "macro_ready", "vol_ready"):
                val = getattr(svc, flag, None)
                (ok if val else note)(f"{flag} = {val}")
            if svc.load_errors:
                for err in svc.load_errors:
                    note(f"load error: {err}")
            note("start the dashboard with:  python -m uvicorn api:app --reload")
            note("then open http://127.0.0.1:8000")
        except Exception as exc:  # noqa: BLE001
            bad(f"serving path FAILED to load: {type(exc).__name__}: {exc}")
            note("TensorFlow is installed, so this is a real failure, not a missing dependency")
            failures += 1

    header("5. Hypothesis registry")
    try:
        import csv
        import glob

        total = drops = 0
        files = sorted(glob.glob("results/*hypothesis_log.csv"))
        for f in files:
            for row in csv.DictReader(open(f, encoding="utf8", errors="replace")):
                total += 1
                drops += "DROP" in (row.get("verdict") or "").upper()
        ok(f"{len(files)} families, {total} registered hypotheses, {drops} recorded as DROP")
    except Exception as exc:  # noqa: BLE001
        bad(f"registry unreadable: {exc}")
        failures += 1

    header("Summary")
    if failures == 0:
        print(f"  {GREEN}Everything required is working.{RESET}")
        print("  Next: notebooks/00_final_report.ipynb  (the narrative report)")
        print("        python -m pytest -q               (the test suite)")
        print("        python -m uvicorn api:app --reload (the live dashboard)")
    else:
        print(f"  {RED}{failures} required check(s) failed.{RESET} See above.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
