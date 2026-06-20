FROM python:3.11-slim

WORKDIR /app

# MetaTrader5 is a Windows-only package used exclusively by the research
# notebook's live MT5 data fetch (Section 2). The containerized Gradio app
# never imports it -- it serves inference from the bundled
# results/eurusd_features.csv history instead -- so it is excluded from the
# image's dependency set rather than failing the Linux build.
COPY requirements.txt .
RUN grep -v -i '^MetaTrader5' requirements.txt > requirements-docker.txt \
    && pip install --no-cache-dir -r requirements-docker.txt

COPY app.py config.json ./
COPY src/ ./src/
COPY models/ ./models/
COPY results/eurusd_features.csv ./results/eurusd_features.csv

EXPOSE 7860
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "app.py"]
