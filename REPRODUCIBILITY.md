# Reproducibility Notes

The project notebook records **Python 3.11.3**, but it does not record the exact package versions used for the original Kaggle run.

The final approach can be reconstructed from the notebook and is implemented in `src/reproduce_best_submission.py`:

- Extra Trees Regressor
- Backward Sequential Feature Selection
- Numeric features: `feat_1`, `feat_6`, `feat_7`, `feat_8`, `feat_9`, `feat_11`, `feat_14`, `feat_15`
- Categorical features: `feat_16`, `feat_18`, `feat_19`
- KNN imputation + standardization for numeric variables
- Most-frequent imputation + one-hot encoding for categorical variables
- 1,000 trees, `max_depth=40`, `max_features=0.6`, `ccp_alpha=0.0001`, no bootstrap
- Prediction averaging over 10 random seeds

A validation run in the packaging environment reproduced the same IDs and overall model behavior, but not every floating-point prediction exactly. This is consistent with the original environment not being fully version-pinned. The official `submissions/predictions_v9.csv` supplied with the project is preserved unchanged and is the file associated with the Kaggle score **0.87054**.
