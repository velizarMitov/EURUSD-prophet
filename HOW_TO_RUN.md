# How to Run This Project (EUR/USD Prophet)

This guide describes how to clone, install, and run the project from scratch. Every step has been verified for real — the repo was cloned into a clean folder, a fresh virtual environment was created, and the server was actually launched before writing this down.

---

## 1. System Requirements

- **Windows** (the project depends on `MetaTrader5`, a Windows-only package; on Linux/macOS, `pip install -r requirements.txt` will fail).
- **Python 3.10+** (tested and working on Python 3.13).
- Internet access (for live data from Yahoo Finance / FRED) — **not required**, though, since the system also works offline using its bundled historical fallback data.

---

## 2. Clone the Repository

```bash
git clone https://github.com/velizarMitov/EURUSD-prophet.git
cd EURUSD-prophet
```

## 3. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs everything: `tensorflow`, `xgboost`, `lightgbm`, `scikit-learn`, `fastapi`, `uvicorn`, `mlflow`, `MetaTrader5`, etc. The first install takes a few minutes.

> **Important:** The trained models (`models/*.pkl`, `*.keras`) and the historical dataset (`results/eurusd_features.csv`) are already committed to the repo. **No training is required** to run the project — it is ready to predict immediately after `pip install`.

## 5. Optional Configuration (`.env`)

The project runs **with zero configuration**, thanks to built-in fallback chains:

| Source | Without configuration, falls back to |
|---|---|
| Live prices | MT5 terminal → Yahoo Finance → bundled history |
| Macro data (FRED yield differential) | FRED API key → public FRED CSV (no key needed) → local cache |

If you want to use an **official FRED API key** or an **MT5 account**, copy the template and fill it in:

```bash
copy .env.example .env
```

then edit `.env`:
```
MT5_LOGIN=
MT5_PASSWORD=
MT5_SERVER=
FRED_API_KEY=
```

Leaving these blank is fine — the system automatically uses the public/fallback sources instead, and this has been confirmed to work without issue.

---

## 6. Running the Application

The application has a single entry point: **`api.py`** (a FastAPI web server). There are three equivalent ways to start it — pick whichever is easiest for you.

### Option A — Double-click (simplest, Windows)

Double-click **`start.bat`** in the project folder. It automatically:
- picks the virtual environment (`.venv` or `venv`) if one exists,
- sets the UTF-8 console encoding,
- frees port 8000 if an old server is still holding it,
- launches the server, and
- **opens the dashboard in your default browser** (`http://127.0.0.1:8000`) once it has started.

A terminal window opens and stays open while the app runs; close it to stop the server.

You can also run it from a terminal:
```bash
start.bat
```

### Option B — Run the file directly

```bash
python api.py
```
`api.py` now has a `__main__` launcher, so running it directly starts the web server (equivalent to Option C, without auto-reload). You can override the host/port with environment variables, e.g. `set PORT=8001` then `python api.py`.

### Option C — Uvicorn with auto-reload (for development)

```bash
python -m uvicorn api:app --reload
```
Use this while editing code — `--reload` restarts the server automatically on file changes.

> **Why did `python api.py` do nothing before?** Earlier the file had no `__main__` block, so running it just imported the module and exited without ever starting the server — it *looked* like the project wouldn't start. That is now fixed: `python api.py` (and `start.bat`) launch the server properly.

Whichever option you choose, then open **http://127.0.0.1:8000** in your browser. Available routes:

| Route | What it does |
|---|---|
| `GET /` | The dashboard (`static/index.html`) — predict button, retrain button, link to history |
| `POST /api/predict` | Runs the real prediction (JSON response) |
| `GET /history` | HTML table comparing every past prediction against the actual market outcome |
| `POST /api/retrain` | Starts model retraining as a background process (15–30 min, non-blocking) |
| `GET /api/retrain/status` | Status of the current retraining run (idle/running/completed/failed) |

The prediction logic itself lives in `src/` (`src/inference.py` and friends); `api.py` is just the web layer on top of it.

### Research Notebook (training & diagnostics, not for serving predictions)
```bash
jupyter notebook notebooks/01_data_preparation.ipynb
```
Contains the full research process — feature engineering, training, diagnostics. "Run All" takes ~15-30 min due to the heavy GridSearch/LSTM cells.

---

## 7. Running the Test Suite

```bash
python -m pytest -q
```
Expected result: **21 passed** (smoke + unit + integration tests).

To run a single test:
```bash
python -m pytest -q -k fetch_yield_differential
```

---

## 8. Common Issues

| Issue | Cause / Fix |
|---|---|
| `OSError: [WinError 10048]` when starting the server | Port 8000 is still held by a previous server. **`start.bat` now frees it automatically before launching.** If starting another way, find the process with `netstat -ano \| findstr :8000` and stop it with `taskkill /PID <pid> /F`, or use a different port: `set PORT=8001` then `python api.py` |
| `UnicodeEncodeError` when running a Python script with Cyrillic/emoji output | The Windows console uses `cp1252`. Set `set PYTHONIOENCODING=utf-8` before the command |
| A notebook cell errors out when running the test suite | Pytest must run from the project **root**, not from `notebooks/`. The Section 20 cells already handle this automatically |
| `pip install -r requirements.txt` fails on Mac/Linux | Expected — `MetaTrader5` has no package for those platforms. This project targets Windows only |

---

## 9. TL;DR

```bash
git clone https://github.com/velizarMitov/EURUSD-prophet.git
cd EURUSD-prophet
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python api.py
```
→ open **http://127.0.0.1:8000** (this runs `api.py` — the full interface with predict, history, and retrain)

Or, on Windows, just **double-click `start.bat`** after the one-time `pip install`.
