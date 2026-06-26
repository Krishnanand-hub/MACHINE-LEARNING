# Machine Learning: Accident-Severity Classification & Cyanobacteria Growth Regression

A two-part supervised machine learning study built for the CSC8635 Machine Learning module (MSc Data Science & AI, Newcastle University). The project applies a disciplined, pipeline-based workflow to two unrelated problems — a large, imbalanced multi-class **classification** task on US road-traffic fatality data, and a **regression** task on cyanobacteria growth-curve parameters — and benchmarks linear, tree-ensemble, and SVM models under cross-validation.

---

## Table of Contents

1. [Aim & Abstract](#aim--abstract)
2. [What This Project Does](#what-this-project-does)
3. [Results & Outcomes](#results--outcomes)
4. [Repository Contents](#repository-contents)
5. [How to Run (Clone & Reproduce)](#how-to-run-clone--reproduce)
6. [Crucial Things to Keep in Mind](#crucial-things-to-keep-in-mind)
7. [Common Errors & Fixes](#common-errors--fixes)
8. [Datasets & Availability](#datasets--availability)
9. [Known Limitations](#known-limitations)

---

## Aim & Abstract

The aim is to demonstrate an end-to-end ML workflow — exploratory analysis, leakage-aware preprocessing, model selection, hyperparameter tuning, and honest evaluation — applied to two contrasting datasets within a single reproducible notebook.

**Dataset 1 (Classification):** Predict the **injury severity** of a road-traffic accident (an 8-class target ranging from *No Injury* to *Fatal Injury*) from the NHTSA Fatality Analysis Reporting System (FARS). The dataset is large (~100,000 records) and heavily imbalanced, which makes raw accuracy a misleading metric and motivates the use of weighted precision, recall, and F1.

**Dataset 2 (Regression):** Predict cyanobacteria growth-curve fit parameters from experimental conditions (e.g. CO₂, light, sucrose ratio, sample count). The task contrasts the limits of linear models against a non-linear tree ensemble on structured scientific data.

The common thread is methodology: every model is wrapped in a scikit-learn `Pipeline` with a `ColumnTransformer` so that scaling and encoding are fit only on training folds, and all comparisons are made under k-fold cross-validation.

---

## What This Project Does

- **Loads and explores** both datasets, surfacing class imbalance (Dataset 1) and feature distributions (Dataset 2) before any modelling.
- **Builds preprocessing pipelines** using `ColumnTransformer` + `StandardScaler` (and `OneHotEncoder` for categoricals) so transformations are leakage-safe across train/test splits.
- **Trains and benchmarks multiple models:**
  - *Classification:* Logistic Regression, Linear SVM, Random Forest, Gradient Boosting.
  - *Regression:* Linear Regression, Ridge, Lasso, Random Forest Regressor.
- **Tunes hyperparameters** with `GridSearchCV` and evaluates with cross-validation rather than a single split.
- **Reports a model-comparison table** with the appropriate metrics for each task (weighted F1 for the imbalanced classifier; R², RMSE, MAE for the regressor).

---

## Results & Outcomes

> All figures below are the **actual outputs** produced by the notebook. Read the [Known Limitations](#known-limitations) section before quoting the classification numbers — they carry an important caveat.

### Dataset 1 — Accident-Severity Classification (5-fold CV, weighted metrics)

| Model | Accuracy | Precision (w) | Recall (w) | F1 (w) |
|---|---|---|---|---|
| Logistic Regression | 0.803 | 0.792 | 0.803 | **0.787** |
| Linear SVM | 0.801 | 0.792 | 0.801 | 0.792 |
| Gradient Boosting | 0.802 | 0.808 | 0.802 | 0.781 |
| Random Forest | 0.779 | 0.771 | 0.779 | 0.774 |

The linear models (Logistic Regression, Linear SVM) marginally outperform the ensembles on weighted F1. Performance is strongly class-dependent: the majority classes are predicted well, while genuine minority classes (e.g. *Possible Injury*) have recall as low as 0.07–0.12.

### Dataset 2 — Cyanobacteria Growth-Parameter Regression

Across the regression targets, the **Random Forest Regressor decisively outperformed all linear models**:

| Model | R² (representative) |
|---|---|
| Random Forest | **0.97 – 0.99** |
| Linear / Ridge / Lasso | 0.21 – 0.62 |

The large gap indicates the relationship between experimental conditions and growth parameters is substantially **non-linear**, which linear models cannot capture but a tree ensemble can.

---

## Repository Contents

| File | Description |
|---|---|
| `CSC8635___ML_Krishnanand.ipynb` | Main notebook containing both analyses end-to-end. |
| `fars.csv` | Dataset 1 — NHTSA FARS person-level accident records (public; see below). |
| `fitting-results.csv` | Dataset 2 — cyanobacteria growth-curve fit parameters (coursework-provided). |
| `README.md` | This file. |

---

## How to Run (Clone & Reproduce)

The notebook was written in **Google Colab** and originally loaded data from Google Drive. You can run it either way.

### Option A — Google Colab (matches the original environment)

1. Open [Google Colab](https://colab.research.google.com/) and upload `CSC8635___ML_Krishnanand.ipynb` (`File → Upload notebook`).
2. Upload `fars.csv` and `fitting-results.csv` to your Google Drive (e.g. into a folder named `ML`).
3. Run the first cell to mount Drive and authorise access:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
4. Confirm the `pd.read_csv(...)` paths in the loading cells point to where you placed the files, then run all cells (`Runtime → Run all`).

### Option B — Local machine (recommended for portability)

```bash
# 1. Clone the repository
git clone https://github.com/Krishnanand-hub/MACHINE-LEARNING.git
cd MACHINE-LEARNING

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

# 4. Launch the notebook
jupyter notebook CSC8635___ML_Krishnanand.ipynb
```

**Before running locally,** you must replace the Colab-specific cells:

- **Remove** the Google Drive mount cell:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- **Change** the hardcoded Drive paths to local relative paths:
  ```python
  # Original (Colab):
  df = pd.read_csv('/content/drive/MyDrive/ML/fars.csv')
  # Local replacement:
  df = pd.read_csv('fars.csv')
  ```

---

## Crucial Things to Keep in Mind

- **Memory & runtime:** `fars.csv` is large (~57 MB, ~100k rows). Loading and training — especially `GridSearchCV` over the SVM and ensemble models — is RAM- and CPU-intensive. On a modest machine some cells take several minutes; the cross-validated SVM fit alone runs ~25–35 s per fold.
- **Run cells in order.** The notebook is stateful: preprocessing pipelines, train/test splits, and fitted objects defined in earlier cells are reused later. Running out of order will raise `NameError`s.
- **The two analyses are independent.** Dataset 1 (classification) and Dataset 2 (regression) do not share variables. Complete one before starting the other.
- **Reproducibility:** splits and models use a fixed `random_state` (42). Keep it set if you want to match the reported numbers.
- **Data is not bundled for local paths.** If you run locally, ensure both CSVs sit in the working directory or update the paths accordingly.

---

## Common Errors & Fixes

| Error / Symptom | Likely Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'google.colab'` | Running the Drive-mount cell outside Colab. | Delete that cell; use a local `pd.read_csv` path (Option B). |
| `FileNotFoundError: .../MyDrive/ML/fars.csv` | Hardcoded Colab path doesn't exist in your environment. | Point `read_csv` to your actual file location. |
| `NameError: name 'X_train' is not defined` | Cells run out of order. | Re-run from the top (`Runtime → Run all`). |
| `MemoryError` / kernel crash on load or `GridSearchCV` | ~100k-row dataset + heavy grid search. | Use a machine with more RAM, sample the data for testing, or reduce the grid / `n_jobs`. |
| `ConvergenceWarning` (LogisticRegression / Lasso / SVM) | Solver hit `max_iter` before converging. | Increase `max_iter`, ensure scaling is applied, or accept it (often non-fatal). |
| Long Drive-mount auth hang in Colab | Authorisation popup not completed. | Re-run the mount cell and complete the Google auth flow. |

---

## Datasets & Availability

### Dataset 1 — FARS (publicly available, public domain)

`fars.csv` is the **NHTSA Fatality Analysis Reporting System** person-level file. It is **U.S. public-domain government data** and contains **no personally identifying information** (no names, addresses, or social security numbers). You can obtain the raw data directly from:

- **NHTSA FTP (all years, 1975–present):** `ftp://ftp.nhtsa.dot.gov/fars/`
- **data.gov:** https://catalog.data.gov/dataset/fatality-analysis-reporting-system-fars
- **Kaggle:** `usdot/nhtsa-traffic-fatalities`
- **Google BigQuery public data:** `bigquery-public-data.nhtsa_traffic_fatalities`

The specific CSV in this repo is a pre-processed subset of person-level records; full coding definitions are in the NHTSA FARS/CRSS Coding and Validation Manual.

### Dataset 2 — Cyanobacteria growth fitting results (coursework-provided)

`fitting-results.csv` contains growth-curve fit parameters (`n_cyanos`, `co2`, `light`, `SucRatio`, `Nsample`, `a`, `mu`, `tau`, `a0`). While the broader topic (cyanobacteria growth curves) is well represented in public repositories such as data.gov (EPA) and published growth-curve datasets, this specific file is **provided as part of the Newcastle University CSC8635 module** and is not redistributed here as an open dataset.

---

## Known Limitations

Stated plainly, because they matter for interpreting the results:

1. **Probable target leakage in the FARS classifier.** The `Fatal_Injury` class is predicted with precision = recall = F1 = **1.00 by every model**. A perfect, model-independent score on one class is a classic signature of a feature that effectively encodes the label. The headline ~80% accuracy is therefore inflated by a trivially separable class and should not be read as genuine predictive skill. A corrected version would audit input features for any direct fatality indicator and remove it before re-training.
2. **Severe class imbalance** depresses minority-class performance (e.g. *Possible Injury* recall ≈ 0.07–0.12). Future work: resampling (SMOTE), class weighting, or threshold tuning.
3. **Coursework scope.** This is an academic study, not a deployed system — there is no serving layer, monitoring, or data-versioning. Treat the metrics as a learning exercise, not production benchmarks.
