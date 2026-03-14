# Boosting Models Analysis

## 1. Overview

This document analyzes the behavior of **boosting-based regression models** on the California Housing dataset.

The goal of this project is to understand how **Gradient Boosting** models learn from data, how their key hyperparameters influence model behavior, and how boosting compares with the tree-based models studied in the previous project.

Experiments in this report focus on:

* Establishing a baseline Gradient Boosting model
* Studying the effect of **learning rate** on model performance
* Analyzing how the number of estimators affects predictive accuracy
* Investigating the role of weak learner complexity
* Understanding which features drive predictions

All evaluations are performed using cross-validation to estimate generalization performance.

---

# 2. Baseline Results

A baseline **Gradient Boosting Regressor** was trained using the same dataset and feature set used in the previous projects. It's model features are as following:

* Max depth: 5
* N estimators: 100
* Learning rate: 0.1

### 2.1 Gradient Boosting

R²: **0.6695**
MAE: **0.4812**
RMSE: **0.6634**

### 2.2 Initial Observations

The Gradient Boosting model explains approximately **67% of the variance** in housing prices.

This performance is very similar to the **Random Forest model** from the previous project:

| Model             | R²    | RMSE  |
| ----------------- | ----- | ----- |
| Decision Tree     | ~0.50 | 0.816 |
| Random Forest     | ~0.67 | 0.666 |
| Gradient Boosting | ~0.67 | 0.663 |

Both ensemble approaches significantly outperform the single Decision Tree and the linear models implemented earlier. This suggests that the dataset contains important **non-linear relationships and feature interactions**.

Although Random Forest and Gradient Boosting achieve similar performance, the two models rely on different learning strategies:

* **Random Forest** reduces variance by averaging many independently trained trees.
* **Gradient Boosting** reduces bias by sequentially correcting prediction errors.

Further experiments will analyze how boosting hyperparameters affect model behavior.

---

# 3. Learning Rate Experiment

To study how the **learning rate** influences boosting performance, multiple models were trained while varying the `learning_rate` parameter.

Each configuration was evaluated using **5-fold cross-validation**, recording the mean R² score and standard deviation across folds. The model features are as following:

* Max depth: 5
* N estimators: 100

### Results:

| Learning Rate | Mean R² | Std R² | RMSE   |
| ------------- | ------- | ------ | ------ |
| 0.01          | 0.4866  | 0.0535 | 0.8013 |
| 0.05          | 0.6640  | 0.0554 | 0.6464 |
| 0.10          | 0.6438  | 0.1147 | 0.6634 |
| 0.20          | 0.6758  | 0.0459 | 0.6360 |
| 0.30          | 0.6362  | 0.0752 | 0.6725 |
| 0.40          | 0.5775  | 0.1726 | 0.7228 |

### Observations

The results show a strong relationship between **learning rate and model performance**.

Very small learning rates such as **0.01** produce substantially worse performance. With such a small step size, each tree only makes a very small correction to the model, and the ensemble of 100 trees is not sufficient to capture the structure of the dataset. This leads to **underfitting**.

Performance improves significantly when the learning rate increases to **0.05 and 0.1**, indicating that the model is able to learn more meaningful corrections at each stage of the boosting process.

Interestingly, the best result in this experiment appears with **learning_rate = 0.2**, which slightly outperforms the baseline configuration and produces the lowest RMSE.

Learning rates of **0.3** and **0.4** cause the model to slightly underperform, leading to lower Mean R² values and higher RMSE and Standard Deviation, suggesting overfitting.

Another notable observation is that **learning_rate = 0.1 shows a larger standard deviation**, suggesting that the model becomes less stable across folds at this configuration.

### Interpretation

The experiment highlights an important property of boosting models: the **learning rate controls how aggressively each tree updates the model**.

* **Very small learning rates** require many trees to learn effectively.
* **Moderate learning rates** allow the model to converge more quickly.
* **Large learning rates** may improve performance initially but can increase the risk of overfitting in larger ensembles.

These results suggest that the interaction between **learning rate and number of estimators** is critical when tuning boosting models.

---

# 4. Number of Estimators Experiment

*(to be completed after experiment)*

This experiment will analyze how increasing the number of boosting iterations affects model performance.

Key questions include:

* How performance evolves as more trees are added
* Whether performance reaches a plateau
* Whether too many estimators lead to overfitting

---

# 5. Weak Learner Complexity

*(to be completed after experiment)*

This section will investigate how the complexity of individual trees influences the behavior of the boosting ensemble.

Possible parameters to study include:

* `max_depth`
* `min_samples_split`
* `min_samples_leaf`

---

# 6. Feature Importance

*(to be completed after analysis)*

Feature importance scores will be extracted from the trained Gradient Boosting model in order to understand which variables contribute most to housing price predictions.

These results will also be compared with the Random Forest feature importance from the previous project.

---

# 7. Discussion

*(to be completed after all experiments)*

This section will summarize the overall behavior of boosting models and compare them with the tree-based ensemble methods studied previously.

---

# 8. Key Takeaways

*(to be completed after experiments)*

---

# 9. Next Steps

Further experiments will explore:

* The relationship between **learning rate and number of estimators**
* The impact of **weak learner complexity**
* Feature importance differences between boosting and Random Forest
* Additional analysis of prediction errors and residuals