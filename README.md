# Sentiment Analysis Project (Python)

A complete, modular sentiment analysis project that classifies text into:
- `positive`
- `negative`
- `neutral`

It includes data preprocessing, model training/evaluation, prediction APIs, CLI interface, visualizations, and model persistence.

## Project Structure

```text
sentiment_analysis_project/
├── app_cli.py
├── app_web.py
├── run_all.ps1
├── requirements.txt
├── README.md
├── data/
│   └── sample_sentiment.csv
├── artifacts/                  # generated after training
└── src/
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── model.py
    ├── predict.py
    ├── preprocessing.py
    ├── train.py
    └── visualize.py
```

## Features Implemented

1. **Data Handling**
   - Loads CSV dataset with `text` and `sentiment` columns
   - Text preprocessing:
     - lowercasing
     - punctuation removal
     - stopword removal
     - tokenization (regex-based)

2. **Model**
   - Traditional ML model: **TF-IDF + Logistic Regression**
   - 3-class sentiment classification (`positive`, `negative`, `neutral`)

3. **Training & Evaluation**
   - Train/test split
   - Accuracy, precision, recall, F1-score
   - Full classification report

4. **Prediction**
   - Function-based prediction API for custom text
   - Real-time interactive prediction mode

5. **Interface**
   - CLI with commands:
     - `train`
     - `predict`
     - `interactive`
   - Streamlit web app (`app_web.py`)

6. **Visualization**
   - Confusion matrix
   - Class distribution bar chart
   - Word clouds per sentiment class

7. **Extras**
   - Save/load trained model (`joblib`)
   - Sample dataset included

---

## Step-by-Step Run Instructions

### 1) Open terminal in project folder

```bash
cd sentiment_analysis_project
```

### 2) (Optional) Create and activate virtual environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Train model

```bash
python app_cli.py train --dataset data/sample_sentiment.csv
```

This creates:
- `artifacts/sentiment_model.joblib`
- `artifacts/metrics.json`
- `artifacts/confusion_matrix.png`
- `artifacts/class_distribution.png`
- `artifacts/wordcloud_positive.png`
- `artifacts/wordcloud_negative.png`
- `artifacts/wordcloud_neutral.png`

### 5) Predict single sentence

```bash
python app_cli.py predict --text "The movie was okay, not great but not bad."
```

### 6) Real-time prediction mode

```bash
python app_cli.py interactive
```

Type text lines and get instant sentiment predictions. Use `exit` to quit.

### 7) Run web app (Streamlit)

```bash
streamlit run app_web.py
```

The web app expects a trained model at `artifacts/sentiment_model.joblib`.  
If missing, train first with:

```bash
python app_cli.py train --dataset data/sample_sentiment.csv
```

### 8) One-command setup + run (Windows PowerShell)

```powershell
.\run_all.ps1
```

Optional custom values:

```powershell
.\run_all.ps1 -Dataset "data/sample_sentiment.csv" -Port 8502
```

---

## Use Your Own Dataset

Use a CSV file with this format:

```csv
text,sentiment
"I love this!",positive
"This is awful.",negative
"It is fine.",neutral
```

Then run:

```bash
python app_cli.py train --dataset /path/to/your_file.csv
```

---

## Optional Deep Learning Extension (Not required)

You can extend this project with:
- LSTM (TensorFlow/PyTorch)
- BERT (`transformers` by Hugging Face)

Current baseline is intentionally lightweight and production-friendly for fast iteration.

