# EURUSD Exchange Rate Predictor: Machine Learning Pipeline

## Project Overview
This repository contains a full end-to-end Machine Learning pipeline predicting the directional closing patterns of the EUR/USD exchange rate. Developed under rigorous academic guidelines for SoftUni ML Learning Final Project, the system explores the **Bias-Variance tradeoff**, implementing robust temporal cross-validation bounds to protect against financial stochastic noise. 

The pipeline formally implements, evaluates, and contrasts:
1. **Advanced Feature Engineering:** Autoregressive lags, bar dynamics, and cyclical datetime encoders.
2. **Deep Learning (LSTM):** Sliding-window sequential topologies exploring implicitly unconstrained modeling.
3. **Gradient Boosting Machine (GBM):** Tree-based iterative residual optimizers capturing feature importance mathematically.

## Project Structure
* **`notebooks/01_data_preparation.ipynb`**: The primary research environment. Contains mathematical formulations in LaTeX, rigorous exploratory data analysis (EDA / ADF / ACF), PCA dimensionality reduction, explicit Multi-Task model construction logic (GBM dual pipeline + Functional API LSTM), hyperparameter tuning, and evaluation plotting.
* **`api.py`**: The FastAPI web server — the application's single entry point. Serves the interactive dashboard (`static/index.html`), the `/api/predict` endpoint, the `/history` prediction-vs-actual page, and the background `/api/retrain` controls.
* **`config.json`**: Centralized hyperparameters and file paths (PCA variance threshold, GBM/LSTM settings, data paths) loaded dynamically by both the notebook and the production scripts.
* **`models/`**: Contains the trained joblib/Keras artifacts (GBM classifier+regressor, Multi-Task LSTM, PCA + scalers).
* **`results/`**: Analytical diagnostic exports including Confusion Matrices, Learning Curves, and the compiled feature subsets.
* **`Dockerfile`**: Optional container definition for running the FastAPI app with strict environment reproducibility (see [Docker Execution](#optional-docker-execution-environment-reproducibility) below).

## Installation & Setup

Ensure Python 3.10+ is natively installed on your operating system.

**1. Clone the repository:**
```bash
git clone https://github.com/velizarMitov/EURUSD-prophet.git
cd EURUSD-prophet
```

## Usage

### 1. Local Execution (Primary Method)
This is the recommended way to run the project — no container runtime required.

**a. Create the virtual environment:**
```bash
python -m venv venv
```

**b. Activate the virtual environment:**
*On Windows:*
```bash
venv\Scripts\activate
```
*On macOS/Linux:*
```bash
source venv/bin/activate
```

**c. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**d. Run the web application:**
```bash
python -m uvicorn api:app --reload
```
*This starts the FastAPI server, by default accessible at `http://127.0.0.1:8000` (the dashboard, the `/api/predict` endpoint, `/history`, and the retrain controls).*

The research notebook uses the same activated environment:
```bash
# Research notebook (EDA, PCA, Multi-Task model training)
jupyter notebook notebooks/01_data_preparation.ipynb
```

### 2. Optional: Running via Docker (Environment Reproducibility)
Containerization is **not required** to run this project — it is provided strictly as an optional MLOps best practice for guaranteeing a byte-identical runtime environment across machines (no "works on my machine" dependency drift). Skip this section unless you specifically need that guarantee.

**a. Build the image:**
```bash
docker build -t eurusd-prophet .
```

**b. Run the container:**
```bash
docker run -p 8000:8000 eurusd-prophet
```
*The dashboard is then reachable on the host at `http://127.0.0.1:8000`, identically to the local execution method above.*

> **Note:** `MetaTrader5` is a Windows-only package and only the first tier of the live-price fallback chain (`src/live_data.py` imports it inside a try/except). The Docker image excludes it and serves inference from Yahoo Finance / the bundled `results/eurusd_features.csv` history instead, so it never fails the Linux build.

## Deployment Model Card
* **Primary Methodology Evaluated:** Gradient Boosting Decision Trees.
* **Usage:** Predicting discrete next-day (+1) temporal bounds based solely on intra-day dynamic geometry matrices.
* **Limitation Notice:** Explicit Dependency on **Global Stationarity** bounds. FX mappings are vulnerable to unmeasured macroeconomic shocks (e.g., Central Bank rate adjustments), immediately bypassing the deterministic tree structure bounds and forcing computational Drift.

## Deployment Model Card: Strengths, Limitations, and Future Work

### 1. Strengths & Pros
- **Gradient Boosting Machines (GBM):** Naturally handles complex, non-linear feature interactions implicitly. Highly robust to extreme market outliers because consecutive data splits rely on ordinality rather than continuous scaled magnitudes.
- **Long Short-Term Memory (LSTM) Networks:** By deploying specialized internal structures (Forget, Input, and Output gates), the LSTM algebraically mitigates error decay. This intelligently controls the empirical time scale of integration, safely capturing long-term temporal dependencies within the EURUSD time series.

### 2. Limitations, Assumptions & Cons
- **Theoretical Assumptions (i.i.d. Violation):** Target scaling mechanisms (percentage log returns) rigorously attempt to enforce stationarity bounds. Despite this, baseline optimizers assume independent and identically distributed (i.i.d.) observations—a constraint heavily violated by the coupled, rapidly transitioning macroeconomic states of global financial markets.
- **Sequential Constraints:** Despite the local structural cell bounds, the vanishing and exploding gradient problem remains a severe persistent theoretical risk when sequence history is iteratively expanded into excessively long look-back windows.

### 3. Future Work & Improvements
- **Bayesian Hyperparameter Optimization:** To navigate the highly complex, non-convex hyperparameter space encountered securely without unconstrained combinatorial grid checks, future automated architectures must definitively deprecate exhaustive boundaries (e.g., `GridSearchCV`) transitioning entirely towards advanced Bayesian surrogate models utilizing libraries like `optuna` or `Spearmint`.
- **Dataset Expansion:** The absolute most statistically effective countermeasure mathematically compressing the generalization gap inside the Bias-Variance tradeoff constraint is scaling sequence volume bounds. Expanding dataset sizing strictly neutralizes and compresses excessive empirical variance.
