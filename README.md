# Dyslexia Risk Prediction using Data Analytics

This repository contains code and resources for predicting dyslexia risk using data analytics and machine learning techniques. The project includes data preprocessing, feature engineering, model training, and a small web interface for demoing the model.

> Note: If you are the repository owner, update the sections below (Dataset, Usage, and Scripts) to reflect the actual files and scripts present in the repository.

## Table of Contents
- [About](#about)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Run in PyCharm (recommended)](#run-in-pycharm-recommended)
  - [Running Scripts from the Command Line](#running-scripts-from-the-command-line)
  - [Run the Webapp](#run-the-webapp)
- [Dataset](#dataset)
- [Modeling & Results](#modeling--results)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

- [Authors / Maintainers](#authors--maintainers)
- [Acknowledgements](#acknowledgements)

## About

Dyslexia Risk Prediction using Data Analytics is a project aimed at identifying children (or participants) at risk of dyslexia by analyzing behavioral, linguistic, and/or educational datasets. The repository contains preprocessing, feature engineering, model training scripts, and artifacts to reproduce results.

## Repository Structure

- README.md
- data/                      # raw and processed datasets (not included in repo if large)
- notebooks/                 # optional Jupyter notebooks for exploration (may be absent)
  - 01_eda.ipynb
  - 02_preprocessing.ipynb
  - 03_modeling.ipynb
- src/                       # Python scripts and modules (project primary code)
- models/                    # saved model artifacts
- results/                   # evaluation results, plots, and reports
- requirements.txt

## Features

- Exploratory data analysis (EDA) to understand distributions and correlations
- Data cleaning and preprocessing pipelines
- Feature engineering for linguistic and behavioral signals
- Multiple classification models (Logistic Regression, Random Forest, XGBoost, etc.)
- Model evaluation with cross-validation and standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- Reproducible scripts demonstrating the workflow (designed to run from PyCharm or CLI)

## Tech Stack

- Language: Python 3.8+
- Data: pandas, numpy
- Modeling: scikit-learn, xgboost (optional)
- Visualization: matplotlib, seaborn

## Getting Started

### Prerequisites

Install Python 3.8 or newer. Recommended to use a virtual environment.

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/harshabhanu7862-hash/Dyslexia-Risk-Prediction-using-Data-Analytics.git
cd Dyslexia-Risk-Prediction-using-Data-Analytics
python -m venv venv
# Activate virtualenv
source venv/bin/activate  # on Windows use `venv\Scripts\activate`

pip install --upgrade pip
pip install -r requirements.txt
```

If there is no requirements.txt, install the common packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyterlab xgboost
```

### Configuration

If the project uses environment variables or a configuration file, create a `.env` or `config.yaml` as required. Example environment variables:

```
DATA_PATH=./data
MODEL_OUTPUT=models/
RANDOM_SEED=42
```

### Run in PyCharm (recommended)

If you developed the project in PyCharm, these steps will help you run and debug the project from the IDE:

1. Open PyCharm and select "Open" > choose the repository folder.
2. Create or select a Python interpreter (use the `venv` created above or a Conda environment).
3. Install any missing packages into that interpreter (PyCharm can prompt to install requirements.txt).
4. Configure Run/Debug configurations:
   - Add a new Python configuration for scripts like `src/train.py`, `src/evaluate.py`, or `app.py` in the Webapp folder.
   - Set working directory to the repository root (or to the Webapp folder for the web app).
   - Add program arguments, e.g. `--config config.yaml` or `--model models/best_model.pkl`.
5. Use the built-in debugger and console to run scripts, inspect variables, and iterate quickly.

Notes:
- If you want to reproduce experiments, create separate configurations for training, evaluation, and the demo webapp.
- You can add PyCharm-specific project files (e.g., `.idea/runConfigurations/`) to the repo if you want to share run configurations with collaborators, but this is optional.

### Running Scripts from the Command Line

If you prefer the terminal (works inside or outside PyCharm), run scripts directly:

```bash
# Example: train the model
python src/train.py --config config.yaml

# Example: evaluate a saved model
python src/evaluate.py --model models/best_model.pkl
```

If `src/` exposes a package entrypoint (e.g., `python -m src.train`), you can use the module form.

### Run the Webapp

A web application is included in the repository at:

```
.idea/New_DyslexiaComputer Games_2026/Webapp
```

To run the webapp locally:

1. Ensure your virtual environment is active and required packages are installed (if the Webapp folder includes its own requirements file, install it):

```bash
# from repo root
cd .idea/New_DyslexiaComputer\ Games_2026/Webapp
# if there is a requirements.txt inside Webapp
pip install -r requirements.txt
```

2. Start the app with:

```bash
python app.py
```

3. The app typically serves on http://localhost:5000 (Flask) or http://localhost:8000 (other frameworks). Check `app.py` for the exact port and configuration.

Notes:
- If the folder name includes spaces, use shell escaping or quotes as shown above.
- If the app requires environment variables (API keys, database URLs), set them before starting the server.

## Dataset

This project expects a dataset with participant-level records and relevant features (e.g., reading scores, phonological awareness, demographic attributes). If the dataset contains sensitive information, keep raw data out of the public repo and provide instructions to recreate processed data from raw sources.

Add a `data/README.md` describing the dataset format, column meanings, and any preprocessing steps required to reconstruct the processed datasets from raw sources.

## Modeling & Results

Document the models you tried and the best-performing approach. Example:

- Logistic Regression baseline
- Random Forest with hyperparameter tuning
- XGBoost with class-weighting and early stopping

Save important artifacts into `models/` and evaluation plots into `results/`.

## Evaluation

Use cross-validation and report metrics with confidence intervals. Include confusion matrices and ROC curves in `results/`.

## Contributing

Contributions are welcome. Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes and push: `git commit -m "Add feature"`
4. Open a Pull Request describing your changes


## Authors / Maintainers

- Project owner: @harshabhanu7862-hash

## Acknowledgements

- Datasets, libraries, and prior research used as reference.
