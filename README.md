# Customer Churn Prediction Project

This project is a small machine learning example focused on predicting customer churn. It uses synthetic customer data to illustrate a simple end-to-end workflow for data generation, exploratory analysis, and model building.

## Overview

The project generates a fictional dataset of customers with attributes such as age, salary, subscription plan, seniority, and usage level. A target variable named `churn` is created based on a simplified rule: customers with low usage and short tenure are considered more likely to leave.

The goal is to build a binary classification model that can estimate whether a customer is likely to churn.

## Project structure

- `generate_data.py` — creates the synthetic dataset and saves it as `churn_data.csv`
- `churn_data.csv` — generated customer data used for analysis and modeling
- `churn_model.ipynb` — notebook containing the exploratory data analysis and model workflow
- `model.pkl` — trained model artifact saved after the notebook workflow
- `venv/` — local Python virtual environment

## Data

The generated dataset includes the following fields:

- `age`
- `salaire` (salary)
- `abonnement` (subscription type)
- `anciennete` (tenure)
- `consommation` (usage score)
- `churn` (target label: 0 = no churn, 1 = churn)

## How it works

1. Run the data generation script to create the dataset.
2. Load the CSV file in the notebook.
3. Explore the data and identify patterns related to customer churn.
4. Train a classification model to predict churn risk.
5. Save the trained model for reuse.

## Run locally

From the project root, run:

```bash
python generate_data.py
```

Then open `churn_model.ipynb` in Jupyter or VS Code and execute the notebook cells.

## Technologies used

- Python
- pandas
- NumPy
- scikit-learn
- Jupyter Notebook

## Purpose

This repository is intended as a lightweight example of an ML workflow in a practical setting. It demonstrates how synthetic data can be generated, analyzed, and used to train a predictive model in a compact, beginner-friendly project.
