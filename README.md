# Dyslexia Risk Prediction using Data Analytics

This repository contains code and resources for predicting dyslexia risk using data analytics and machine learning techniques. The project includes data preprocessing, feature engineering, model training, evaluation, and example notebooks to reproduce experiments.

> Note: If you are the repository owner, update the sections below (Dataset, Usage, and Scripts) to reflect the actual files and notebooks present in the repository.

## Table of Contents
- [About](#about)
- [Repository Structure](#repository-structure)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running Notebooks and Scripts](#running-notebooks-and-scripts)
  - [Run the Webapp](#run-the-webapp)
- [Dataset](#dataset)
- [Modeling & Results](#modeling--results)
- [Evaluation](#evaluation)
- [Contributing](#contributing)

- [Authors / Maintainers](#authors--maintainers)
- [Acknowledgements](#acknowledgements)

## About

Dyslexia Risk Prediction using Data Analytics is a project aimed at identifying children (or participants) at risk of dyslexia by analyzing behavioral, linguistic, and/or educational datasets. The repository shows the end-to-end pipeline from data cleaning and exploratory analysis to building and evaluating machine learning models.

## Repository Structure


- README.md
- data/                      # raw and processed datasets (not included in repo if large)
- notebooks/                 # Jupyter notebooks for EDA, preprocessing, and modeling
  - 01_eda.ipynb
  - 02_preprocessing.ipynb
  - 03_modeling.ipynb
- src/                       # Python scripts and modules (if present)
- models/                    # saved model artifacts
- results/                   # evaluation results, plots, and reports
- requirements.txt

## Features

- Exploratory data analysis (EDA) to understand distributions and correlations
- Data cleaning and preprocessing pipelines
- Feature engineering for linguistic and behavioral signals
- Multiple classification models (Logistic Regression, Random Forest, XGBoost, etc.)
- Model evaluation with cross-validation and standard metrics (accuracy, precision, recall, F1, ROC-AUC)
- Reproducible Jupyter notebooks demonstrating the workflow

## Tech Stack

- Language: Python 3.8+
- Data: pandas, numpy
- Modeling: scikit-learn, xgboost (optional)
- Visualization: matplotlib, seaborn
- Notebooks: Jupyter

## Getting Started

### Prerequisites

Install Python 3.8 or newer. Recommended to use a virtual environment.

### Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/harshabhanu7862-hash/Dyslexia-Risk-Prediction-using-Data-Analytics.git
cd Dyslexia-Risk-Prediction-using-Data-Analytics
python -m venv venv
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

### Running Notebooks and Scripts

Start Jupyter Lab / Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

Open the notebooks in `notebooks/` and run cells in order. Typical workflow:

1. 01_eda.ipynb — explore the data and visualize distributions
2. 02_preprocessing.ipynb — clean the data and create features
3. 03_modeling.ipynb — train models and evaluate

If there are CLI scripts in `src/`, run them like:

```bash
python src/train.py --config config.yaml
python src/evaluate.py --model models/best_model.pkl
```

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

This project expects a dataset with participant-level records and relevant features (e.g., reading scores, phonological awareness, demographic attributes). If the dataset contains sensitive information, do not commit it to the repo — add it to `.gitignore` and provide instructions for obtaining the data.

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
