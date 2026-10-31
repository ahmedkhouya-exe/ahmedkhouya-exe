# Smart Data Analyzer

A machine-learning web application that explores scikit-learn's built-in Wine dataset and evaluates a Random Forest classifier through an interactive Streamlit dashboard.

Built with Python, pandas, scikit-learn, Streamlit, and pytest.

> **Note:** This project is for educational purposes only. It is not intended for real-world decisions.

## Features

- **Dataset exploration** — preview samples and explore class distributions.
- **Machine learning** — train a Random Forest classifier on the Wine dataset.
- **Configurable training** — adjust the number of trees and the train/test split.
- **Model evaluation** — view accuracy and a detailed classification report.
- **Confusion matrix** — compare actual and predicted classes.
- **Feature importance** — visualize which input features contribute most to the model.
- **Automated tests** — verify core data-loading and model-evaluation functionality.

## Tech Stack

- Python
- pandas
- scikit-learn
- Streamlit
- pytest

## Project Structure

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

## Getting Started

### Prerequisites

- Python 3.9 or newer
- pip

### 1. Clone the repository

```bash
git clone https://github.com/ahmedkhouya-exe/ahmedkhouya-exe.git
cd ahmedkhouya-exe/smart-data-analyzer
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**macOS / Linux**
```bash
source .venv/bin/activate
```

**Windows**
```bash
.venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Launch the application

```bash
streamlit run src/app.py
```

Streamlit will provide a local URL where you can open the dashboard in your browser.

## Run Tests

With the virtual environment activated, run:

```bash
python -m pytest tests -v
```

## Model Evaluation

The application evaluates the classifier on a held-out test set and displays:

- Accuracy
- Precision, recall, and F1-score
- A confusion matrix
- Feature importance

Results depend on the selected train/test split and model configuration. A high score on this small dataset does not guarantee performance on new, real-world data.

## Limitations

- Uses scikit-learn's built-in Wine dataset rather than a large, real-world dataset.
- Results can vary depending on the train/test split and model settings.
- Feature importance indicates the model's relative use of features; it does not establish causation.
- Further validation would be required before applying this approach to real-world problems.

## Author

**Ahmed Khouya**

GitHub: [@ahmedkhouya-exe](https://github.com/ahmedkhouya-exe)
