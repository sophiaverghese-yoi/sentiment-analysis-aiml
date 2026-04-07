param(
    [string]$Dataset = "data/sample_sentiment.csv",
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"

Write-Host "==> Installing dependencies..."
python -m pip install -r requirements.txt

Write-Host "==> Training sentiment model..."
python app_cli.py train --dataset $Dataset

Write-Host "==> Launching Streamlit app on port $Port ..."
python -m streamlit run app_web.py --server.port $Port

