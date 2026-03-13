# ML Housing Price Predictor – Boosting Models

A structured machine learning project focused on understanding and evaluating **gradient boosting regression models** through a disciplined, reproducible training workflow.

This project builds upon previous work with linear models and tree-based ensembles by introducing **boosting**, one of the most powerful techniques for tabular machine learning. The goal is to understand how boosting improves predictive performance, how its hyperparameters influence model behavior, and how it compares to previous models such as linear regression and random forests.

The emphasis of this project is **model intuition, experimentation discipline, and systematic analysis of boosting behavior** rather than system complexity or large-scale feature engineering.

---

# Project Objectives

This project aims to implement a production-style machine learning workflow that:

1. Loads structured housing data
2. Trains and evaluates gradient boosting regression models
3. Studies the effect of key boosting hyperparameters
4. Compares boosting models against previous approaches
5. Selects the best-performing model programmatically
6. Saves a fully self-contained trained model artifact
7. Performs inference with strict feature validation

The project prioritizes **understanding model behavior, rigorous evaluation, and reproducibility**.

---

# Core Learning Goals

This project is designed to strengthen understanding of several key machine learning concepts.

## Boosting

Boosting is an ensemble technique where models are trained **sequentially**, with each new model focusing on correcting the errors made by the previous ones.

Instead of averaging many independent models (as in Random Forest), boosting gradually builds a strong predictor by **iteratively improving residual errors**.

This process allows boosting models to achieve very strong predictive performance on tabular datasets.

---

## Sequential Learning

Unlike bagging-based ensembles such as Random Forests, boosting models learn **in stages**.

Each new tree is trained to predict the **residual error** of the current model. Over many iterations, the model gradually improves its predictions.

This sequential process introduces a new set of hyperparameters that control learning dynamics.

---

## Learning Rate and Model Updates

Boosting models include a **learning rate** parameter that controls how much each tree contributes to the final prediction.

A smaller learning rate means:

* slower learning
* more trees required
* potentially better generalization

Understanding this tradeoff is a core goal of this project.

---

## Number of Trees (Estimators)

Boosting models rely on many shallow trees trained sequentially.

The `n_estimators` parameter determines how many trees are added to the ensemble. Increasing this value can improve performance but may also increase the risk of overfitting.

Studying this relationship provides insight into **model complexity in boosting systems**.

---

## Feature Importance

Like other tree-based models, gradient boosting provides **feature importance estimates** that reveal which variables contribute most to predictions.

Comparing these results with previous projects helps understand how different model families interpret the same dataset.

---

# Candidate Models

This project evaluates the following model:

* `GradientBoostingRegressor`

This implementation is provided by **scikit-learn** and serves as a clear and interpretable introduction to boosting methods.

Future projects may explore more advanced boosting libraries such as **XGBoost**, **LightGBM**, or **CatBoost**.

---

# Training Workflow

Model training follows a structured workflow:

1. Load dataset
2. Separate features and target variable
3. Build candidate models
4. Evaluate models using cross-validation
5. Compare mean and variance of performance metrics
6. Select the best-performing model
7. Train the final model on the full dataset
8. Save a structured model artifact containing:

* trained model
* feature schema
* metadata
* training configuration

This workflow emphasizes **reproducibility and consistent evaluation across experiments**.

---

# Analysis and Experiments

The project includes dedicated analysis scripts exploring the behavior of boosting models.

Experiments include:

* Baseline Gradient Boosting performance
* Learning rate experiments
* Number of estimators experiments
* Feature importance analysis
* Performance comparison with previous models:

  * Linear Regression
  * Ridge Regression
  * Decision Tree
  * Random Forest

Experiment outputs and notes are stored in the `reports/` directory.

---

# Project Structure

```
ml-housing-predictor-boosting/
├── data/
├── input/
├── reports/
├── scripts/
├── train.py
├── predict.py
├── ml_engine.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Scope

This project intentionally focuses on **classical machine learning using gradient boosting models**.

Out of scope:

* Deep learning
* Neural networks
* API deployment
* Distributed training
* Advanced boosting libraries such as XGBoost or LightGBM

These topics may be explored in later projects.

---

# Requirements

* Python 3.9+
* pandas
* numpy
* scikit-learn

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Philosophy

This project prioritizes:

* Understanding over speed
* Controlled experimentation over blind hyperparameter tuning
* Reproducibility over ad-hoc experimentation
* Engineering discipline over notebook-style workflows

The goal is to develop **deep intuition about boosting models and their behavior in real-world tabular prediction tasks**.