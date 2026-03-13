"""
This script contains functions to analyze the performance of the decision tree and random forest models, including error analysis, feature importance, and visualizations.
The functions can be called to generate reports and visualizations that help understand the strengths and weaknesses of each model, as well as the importance of different features in the dataset.
The analysis includes:
- Baseline model evaluation using cross-validation predictions
- Error analysis to identify patterns in the prediction errors
- Experimenting with different tree depths to find the optimal depth for the decision tree model
- Visualizing the structure of a decision tree
- Analyzing feature importance in the random forest model
To run the analysis, simply execute this script. The generated reports and visualizations will be saved in the "reports" directory.
"""

from numpy import sqrt
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
from ml_engine import load_dataset, build_models, add_features
from sklearn.tree import plot_tree

#Baseline model evaluation using cross-validation predictions
X, y = load_dataset("data/california_housing.csv")
model_boost = build_models().get("gradient_boosting")

def analyze_models(X : pd.DataFrame, y : pd.Series, model_boost) -> None:
    pred_boost = cross_val_predict(model_boost, X, y, cv=5)

    r2_boost = r2_score(y, pred_boost)
    rmse_boost = sqrt(mean_squared_error(y, pred_boost))
    mae_boost = mean_absolute_error(y, pred_boost)

    errors_boost = pred_boost - y
    abs_errors_boost = abs(errors_boost)

    worst_boost_idx = abs_errors_boost.sort_values(ascending=False).head(10).index

    print(f"Gradient Boosting - R²: {r2_boost:.4f}, RMSE: {rmse_boost:.4f}, MAE: {mae_boost:.4f}")
    print("\nGradient Boosting error stats:")
    print("mean:", errors_boost.mean())
    print("std:", errors_boost.std())
    print("min:", errors_boost.min())
    print("max:", errors_boost.max())

    print("\nWorst Gradient Boosting predictions:")
    print(X.loc[worst_boost_idx])
    print(y.loc[worst_boost_idx])


#Baseline model evaluation
#analyze_models(X, y, model_boost)