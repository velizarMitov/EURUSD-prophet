FROM python:3.11-slim

WORKDIR /app

# MetaTrader5 is a Windows-only package and is only the first tier of the live
# price fallback chain in src/live_data.py, which imports it inside a try/except.
# On Linux the container therefore falls back to Yahoo Finance / the bundled
# history snapshot, so MetaTrader5 is excluded here rather than failing the build.
COPY requirements.txt .
RUN grep -v -i '^MetaTrader5' requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY api.py config.json ./
COPY src/ ./src/
COPY static/ ./static/
COPY models/ ./models/
COPY results/eurusd_features.csv ./results/eurusd_features.csv

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
