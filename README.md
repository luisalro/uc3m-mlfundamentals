[README.md](https://github.com/user-attachments/files/31785812/README.md)
# Machine Learning Fundamentals — Kaggle Competition

[![Kaggle](https://img.shields.io/badge/Kaggle-Competition-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/fundamentos-de-aprendizaje-automatico-abril-2026)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Final Rank](https://img.shields.io/badge/Final%20Rank-1st-success)
![Private R2](https://img.shields.io/badge/Private%20R%C2%B2-0.87054-success)

A complete regression workflow developed for the **Machine Learning Fundamentals — April 2026** Kaggle competition.

The project covers the full modeling cycle: exploratory analysis, preprocessing, feature selection, model comparison, hyperparameter tuning and ensembling. The final submission finished **1st overall** on the final Private Leaderboard with an **R² score of 0.87054**.

---

## Highlights

| | Result |
|---|---|
| **Final position** | 🏆 1st place |
| **Public Leaderboard** | `R² = 0.86424` |
| **Private Leaderboard** | **`R² = 0.87054`** |
| **Best model family** | Extra Trees Regressor |
| **Feature selection** | Backward Sequential Feature Selection |
| **Final ensemble** | Average of 10 random seeds |
| **Evaluation metric** | Coefficient of determination (`R²`) |

The final ranking was based on the **Private Leaderboard**, which used approximately 55% of the test set.

---

## Competition

The task is a supervised **regression problem** built from anonymized tabular data.

Given a training dataset containing numerical and categorical features, the objective is to learn a model capable of predicting a continuous `target` value for unseen observations in the test set.

Kaggle evaluates submissions using the coefficient of determination:

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Higher values indicate better predictive performance.

---

## Final leaderboard result

### Public Leaderboard

The public score was calculated using approximately 45% of the hidden test data.

![Public leaderboard — Grupo 4 in 1st place](assets/public-leaderboard.png)

**Score:** `0.86424`  
**Position:** `1st`

### Private Leaderboard

The private score was calculated using the remaining approximately 55% of the hidden test data and determined the final competition ranking.

![Private leaderboard — Grupo 4 in 1st place](assets/private-leaderboard.png)

**Final score:** `0.87054`  
**Final position:** **1st overall**

The exact file associated with this result is available at [`submissions/predictions_v9.csv`](submissions/predictions_v9.csv).

---

## Solution overview

The final solution is the result of an iterative experimentation process rather than a single model run.

The strongest pipeline combined:

1. missing-value treatment;
2. separate preprocessing for numerical and categorical variables;
3. backward sequential feature selection;
4. a tuned `ExtraTreesRegressor`;
5. prediction averaging across 10 random seeds.

```text
train_set.csv
      │
      ▼
Data exploration
      │
      ▼
Preprocessing
      │
      ├── Numeric ───── KNN imputation + scaling
      │
      └── Categorical ─ Most-frequent imputation + one-hot encoding
      │
      ▼
Backward Sequential Feature Selection
      │
      ▼
Tuned Extra Trees Regressor
      │
      ▼
10-seed prediction averaging
      │
      ▼
predictions_v9.csv
      │
      ▼
Private Leaderboard R² = 0.87054
```

---

## Data

The original competition files are stored in [`data/`](data/).

| File | Shape | Description |
|---|---:|---|
| `train_set.csv` | 3,504 × 22 | Training observations, `ID`, features and `target` |
| `test_set.csv` | 4,833 × 21 | Test observations and `ID` |
| `dumb_submission.csv` | 4,833 × 2 | Example submission structure |

The dataset contains **20 anonymized features**, named from `feat_0` to `feat_19`.

The categorical variables used by the final pipeline are:

- `feat_16`
- `feat_18`
- `feat_19`

The remaining variables are numerical.

---

## Preprocessing

Different preprocessing strategies were tested during development. The final workflow uses separate transformations by feature type.

### Numerical features

```python
KNNImputer(n_neighbors=5)
StandardScaler()
```

### Categorical features

```python
SimpleImputer(strategy="most_frequent")
OneHotEncoder(handle_unknown="ignore")
```

This keeps missing-value treatment and encoding inside the modeling pipeline and avoids manually transforming the test set independently.

---

## Feature selection

Several feature-selection strategies were explored, including mutual information, tree-based feature importance and sequential selection.

The best cross-validation result came from **Backward Sequential Feature Selection**.

The selected numerical features were:

```text
feat_1
feat_6
feat_7
feat_8
feat_9
feat_11
feat_14
feat_15
```

Together with the categorical variables, these features formed the input to the final Extra Trees model.

---

## Final model

The winning submission uses a tuned `ExtraTreesRegressor`.

```python
ExtraTreesRegressor(
    n_estimators=1000,
    max_depth=40,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=0.6,
    bootstrap=False,
    ccp_alpha=0.0001,
)
```

### Why Extra Trees?

Among the tested model families, Extra Trees consistently provided the strongest validation performance for this dataset. It also worked particularly well after reducing the numerical feature space with backward selection.

The final prediction was not generated from only one fitted estimator. The same tuned configuration was trained with the following random seeds:

```text
42, 0, 7, 13, 99, 123, 256, 512, 777, 999
```

The ten prediction vectors were then averaged to produce the final submission.

---

## Experiments

The notebook contains the complete experimentation history. Models and techniques tested include:

- Ridge Regression
- Lasso Regression
- Random Forest
- Extra Trees
- Support Vector Regression
- Voting ensembles
- Stacking
- Yeo-Johnson target transformation
- KNN imputation
- Iterative imputation
- Gaussian Mixture Models
- Mutual Information
- Random Forest feature importance
- Forward Sequential Feature Selection
- Backward Sequential Feature Selection
- Randomized hyperparameter search
- Multi-seed averaging

A selection of 5-fold cross-validation results is shown below.

| Experiment | Mean CV R² |
|---|---:|
| Ridge + target transformation | 0.5523 |
| Random Forest + target transformation | 0.7799 |
| Extra Trees + target transformation | 0.8074 |
| Extra Trees baseline | 0.8093 |
| Extra Trees + Forward SFS | 0.8284 |
| **Extra Trees + Backward SFS** | **0.8323** |
| Tuned Extra Trees + Backward SFS | 0.8319 |
| 10-seed mean CV | 0.8318 |

> Cross-validation and Kaggle leaderboard scores are computed on different partitions, so they are not expected to be identical.

---

## Repository structure

```text
machine-learning-fundamentals-kaggle-2026/
│
├── README.md
├── RESULTS.md
├── REPRODUCIBILITY.md
├── requirements.txt
├── .gitignore
│
├── assets/
│   ├── public-leaderboard.png
│   └── private-leaderboard.png
│
├── data/
│   ├── train_set.csv
│   ├── test_set.csv
│   └── dumb_submission.csv
│
├── notebooks/
│   └── kaggle_model_ml.ipynb
│
├── src/
│   └── reproduce_best_submission.py
│
└── submissions/
    └── predictions_v9.csv
```

---

## Reproduce the solution

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd machine-learning-fundamentals-kaggle-2026
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate the environment.

**Linux / macOS**

```bash
source .venv/bin/activate
```

**Windows**

```powershell
.venv\Scripts\activate
```

### 3. Install the dependencies

```bash
pip install -r requirements.txt
```

### 4. Reproduce the final pipeline

```bash
python src/reproduce_best_submission.py
```

The generated predictions are saved to:

```text
submissions/predictions_reproduced.csv
```

For the complete research process, open the notebook:

```text
notebooks/kaggle_model_ml.ipynb
```

---

## Submission format

Kaggle expects a CSV file with one prediction for each test observation:

```csv
ID,target
0,-191.2249
1,-141.5563
2,-320.8883
```

The winning [`predictions_v9.csv`](submissions/predictions_v9.csv) contains **4,833 predictions**, unique IDs and no missing target values.

---

## Reproducibility note

The original notebook metadata records Python `3.11.3`, but it does not preserve the exact versions of every library used when `predictions_v9.csv` was generated.

Small differences in scikit-learn versions can affect tree ensembles and random-number behavior. For that reason:

- [`submissions/predictions_v9.csv`](submissions/predictions_v9.csv) is the authoritative submission associated with the final Kaggle score;
- [`src/reproduce_best_submission.py`](src/reproduce_best_submission.py) reproduces the documented modeling strategy;
- [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) contains additional implementation notes.

---

## Key takeaway

This competition was a useful example of how **systematic experimentation can matter more than model complexity**.

The largest improvements did not come from moving to increasingly complicated algorithms, but from combining a strong tree ensemble with careful feature selection, disciplined validation and low-cost ensembling across random seeds.

The result was a final **Private Leaderboard R² of `0.87054` and 1st place overall**.

---

## Acknowledgements

This repository was created from the work carried out for the **Machine Learning Fundamentals** course competition hosted on Kaggle.

Competition data, evaluation rules and challenge materials remain subject to the original Kaggle competition terms.
