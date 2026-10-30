# Smart Data Analyzer

A beginner-friendly machine-learning demo that classifies samples from scikit-learn's built-in Wine dataset. It provides a Streamlit interface for exploring the data and viewing model evaluation results.

> Educational demonstration only. It is not intended for decisions about real-world products or people.

## Features
- Loads a built-in dataset (no separate download required)
- Shows a dataset preview and class distribution
- Trains a Random Forest classifier using a stratified train/test split
- Reports accuracy and a classification report on held-out test data
- Lets users adjust the number of trees and test-set size
- Displays the most important model features

## Tech stack
Python, pandas, scikit-learn, Streamlit, pytest

## Setup
Requires Python 3.10 or newer.

```bash
cd smart-data-analyzer
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Run the app
```bash
streamlit run src/app.py
```

## Run tests
```bash
python -m pytest
```

## Project structure
```text
smart-data-analyzer/
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── model.py
├── tests/
│   └── test_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Limitations
- This is a small, clean, built-in teaching dataset and does not represent the complexity of most production data.
- Accuracy varies with the chosen train/test split and model settings.
- Model performance should be evaluated using appropriate metrics and domain context before any real-world use.
