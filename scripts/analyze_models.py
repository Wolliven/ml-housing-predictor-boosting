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
from ml_engine import load_dataset, build_boost

#Baseline model evaluation using cross-validation predictions
X, y = load_dataset("data/california_housing.csv")
model_boost = build_boost()

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

#learning rate experiment with table of results
def learning_rate_experiment(X : pd.DataFrame, y : pd.Series) -> None:
    learning_rates = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
    results = []
    for lr in learning_rates:
        model_boost = build_boost(learning_rate=lr)
        scores_boost = cross_val_score(model_boost, X, y, cv=5, scoring="r2")
        mean_boost = scores_boost.mean()
        rmse_boost = sqrt(mean_squared_error(y, cross_val_predict(model_boost, X, y, cv=5)))
        std_boost = scores_boost.std()
        results.append((lr, mean_boost, std_boost, rmse_boost))
    print("Learning Rate Experiment Results:")
    print("Learning Rate | Mean R² | Std R² | RMSE")
    for lr, mean_boost, std_boost, rmse_boost in results:
        print(f"{lr:<14} | {mean_boost:.4f} | {std_boost:.4f} | {rmse_boost:.4f}")


#n_estimators experiment with table of results
def n_estimators_experiment(X : pd.DataFrame, y : pd.Series) -> None:
    n_estimators_list = [50, 100, 200, 300, 400, 500]
    results = []
    for n in n_estimators_list:
        model_boost = build_boost(n_estimators=n, learning_rate=0.05)
        scores_boost = cross_val_score(model_boost, X, y, cv=5, scoring="r2")
        mean_boost = scores_boost.mean()
        rmse_boost = sqrt(mean_squared_error(y, cross_val_predict(model_boost, X, y, cv=5)))
        std_boost = scores_boost.std()
        results.append((n, mean_boost, std_boost, rmse_boost))
    print("N Estimators Experiment Results:")
    print("N Estimators | Mean R² | Std R² | RMSE")
    for n, mean_boost, std_boost, rmse_boost in results:
        print(f"{n:<13} | {mean_boost:.4f} | {std_boost:.4f} | {rmse_boost:.4f}")

#n_estimators_experiment(X, y)

def weak_learner_analysis(X : pd.DataFrame, y : pd.Series) -> None:
    max_depth_list = [1, 2, 3, 4, 5, 6]
    results = []
    for max_depth in max_depth_list:
        model_boost = build_boost(n_estimators=100, learning_rate=0.05, max_depth=max_depth)
        scores_boost = cross_val_score(model_boost, X, y, cv=5, scoring="r2")
        mean_boost = scores_boost.mean()
        rmse_boost = sqrt(mean_squared_error(y, cross_val_predict(model_boost, X, y, cv=5)))
        std_boost = scores_boost.std()
        results.append((max_depth, mean_boost, std_boost, rmse_boost))

    print("Weak Learner Analysis Results:")
    print("Max Depth | Mean R² | Std R² | RMSE")
    for max_depth, mean_boost, std_boost, rmse_boost in results:
        print(f"{max_depth:<10} | {mean_boost:.4f} | {std_boost:.4f} | {rmse_boost:.4f}")

#weak_learner_analysis(X, y)

def feature_importance_analysis(X : pd.DataFrame, y : pd.Series) -> None:
    model_boost = build_boost()
    model_boost.fit(X, y)
    importances = model_boost.feature_importances_
    feature_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)
    plt.figure(figsize=(8,5))
    feature_importance.plot(kind="bar")
    plt.title("Feature Importance - Gradient Boosting")
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig("reports/boosting_feature_importance.png", dpi=300)
    plt.show()

#feature_importance_analysis(X, y)