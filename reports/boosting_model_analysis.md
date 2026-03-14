# Baseline Analysis – Boosting Model

## Overview

This analysis evaluates the baseline performance of a **Gradient Boosting regression model** on the California Housing dataset.

The goal is to understand how a boosting-based ensemble compares with the tree-based models studied in the previous project, particularly the **Decision Tree** and **Random Forest** models.

Evaluation is performed using cross-validation predictions to estimate generalization performance.

---

# Metrics

## Gradient Boosting

R²: 0.6695
MAE: 0.4812
RMSE: 0.6634

---

# Error Statistics

## Gradient Boosting

mean error: 0.0834
standard deviation: 0.6582
max error: 3.2574
min error: -3.7339

---

# Worst Prediction Samples

The largest prediction errors occur in districts with unusual or extreme feature values.

Several examples show census blocks with very large `AveRooms` values (e.g. `AveRooms > 100`), which likely represent atypical aggregated reporting artifacts rather than typical residential districts.

Other problematic rows involve districts with very small population counts or unusual occupancy ratios. These irregular feature patterns appear to create conditions where the model struggles to estimate housing prices accurately.

Even though boosting combines many trees sequentially, extreme observations can still generate large prediction errors when they differ significantly from the majority of the dataset.

---

# Observations

### Gradient Boosting Performance

The Gradient Boosting model explains approximately **67% of the variance** in housing prices.

This performance is very similar to the Random Forest model from the previous project and represents a clear improvement over the single Decision Tree baseline.

The RMSE and MAE values also indicate relatively stable predictions, suggesting that the ensemble of sequential trees successfully captures important non-linear patterns in the data.

---

### Comparison with Previous Tree Models

From the previous project:

| Model             | R²    | RMSE  |
| ----------------- | ----- | ----- |
| Decision Tree     | ~0.50 | 0.816 |
| Random Forest     | ~0.67 | 0.666 |
| Gradient Boosting | ~0.67 | 0.663 |

Gradient Boosting achieves performance comparable to Random Forest, with slightly lower RMSE.

This suggests that both ensemble approaches effectively model the dataset, though they rely on different learning strategies:

* **Random Forest** reduces variance by averaging many independently trained trees.
* **Gradient Boosting** reduces bias by sequentially correcting prediction errors.

---

### Error Distribution

The mean prediction error is close to zero, indicating that the model does not exhibit a strong global bias toward overprediction or underprediction.

The standard deviation of the errors is slightly lower than that of the Random Forest model, suggesting that predictions are similarly stable overall.

However, large individual errors still occur for districts with unusual feature values, particularly when variables such as `AveRooms`, `Population`, or `AveOccup` take extreme values.

These outliers appear consistently among the worst predictions across multiple model types.

---

# Key Takeaways

1. Gradient Boosting achieves predictive performance comparable to Random Forest, explaining roughly **67% of the variance** in housing prices.
2. The boosting ensemble produces stable predictions with relatively low RMSE and MAE.
3. Sequential error correction allows the model to capture complex relationships in the dataset.
4. Extreme feature values continue to appear among the largest prediction errors, suggesting that irregular census blocks remain difficult to model.

---

# Next Steps

Further analysis will focus on understanding the internal behavior of boosting models.

Possible experiments include:

* Learning rate experiments to study how aggressively the model updates predictions
* Number of estimators experiments to analyze how performance evolves as trees are added
* Tree depth experiments to examine how weak learner complexity affects generalization
* Feature importance analysis to identify the most influential variables

These experiments should provide deeper insight into how boosting models learn and how their hyperparameters influence predictive performance.