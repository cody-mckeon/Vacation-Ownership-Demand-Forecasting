# Vacation Ownership Demand Forecasting and Upsell Recommendation System

## Overview
Predict Weekly unit occupancy and recommend ancillary upsell packages (spa, dining, excursions) to improve revenue forecasting and operational planning for vacation ownership properties.

## Features
- Data ingestion and feature engineering pipeline with DVC
- Time-Series forecasting model using XGBoost
- Experiment tracking and model registry with MLflow
- CI/CD automation via GitHub Actions
- REST API for serving predictions (FastAPI)
- Interactive dashboard (Streamlit or Google Data Studio)



- vacation-ownership-demand-forecast/
- ├── data/                      
- │   ├── raw/                   
- │   │   └── README.md          # instructions for adding raw data (CSV/SQL exports)
- │   └── processed/             
- │       └── README.md          # instructions for processed feature files (parquet/csv)
- ├── feature_factory/           
- │   ├── feature_engineering.ipynb # Colab notebook: data cleansing & feature creation
- │   └── scripts/               
- │       └── build_features.py  # Python script to replicate notebook logic
- ├── models/                    
- │   ├── train_model.ipynb      # Colab notebook: model training & MLflow tracking
- │   └── train.py               # Python script for headless training
- ├── serve/                     
- │   └── app.py                 # FastAPI or Streamlit app for serving predictions
- ├── .github/                   
- │   └── workflows/             
- │       ├── ci.yml             # GitHub Actions: lint, test pipeline
- │       └── retrain.yml        # GitHub Actions: scheduled retraining job
- ├── docs/                      
- │   ├── architecture.md        # Markdown: system architecture & diagrams
- │   └── data_dictionary.md     # Markdown: feature definitions
- ├── slides/                    
- │   └── demand_forecast_presentation.pdf # Exported slide deck for stakeholders
- ├── dvc.yaml                   # DVC pipeline definitions for data and features
- ├── README.md                  # Project overview, setup instructions, goals
- ├── requirements.txt           # pinned Python dependencies
- └── .gitignore                 # ignore patterns for Python, DVC, data files
