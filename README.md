[README.md](https://github.com/user-attachments/files/26289793/README.md)
# Multi-Label Email Classification

> NCI — Engineering and Evaluating Artificial Intelligence (CA1)  
> Continuous Assessment 1 · Chained Multi-Output Classification

A modular Python pipeline for multi-label email classification using **Design Choice 1: Chained Multi-Output Classification**. The system classifies customer support emails across three dependent label types (Type 2, Type 3, Type 4) and benchmarks five ML models side-by-side across all chain levels.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [How to Run](#how-to-run)
- [Models](#models)
- [Results](#results)
- [How to Add a New Model](#how-to-add-a-new-model)
- [Design Principles](#design-principles)

---

## Project Structure

```
.
├── main.py                     # Controller — entry point
├── Config.py                   # Shared constants and column names
├── preprocess.py               # Data loading, deduplication, noise removal
├── embeddings.py               # TF-IDF text vectorisation
├── utils.py                    # Chain label builder (build_chain_labels)
├── data/
│   ├── AppGallery.csv          # Email dataset 1
│   └── Purchasing.csv          # Email dataset 2
├── modelling/
│   ├── data_model.py           # Data class — encapsulates train/test splits
│   └── modelling.py            # Model runner and comparison table printer
└── model/
    ├── base.py                 # Abstract base class (BaseModel)
    ├── randomforest.py         # Random Forest
    ├── svm.py                  # Support Vector Machine (LinearSVC)
    ├── logistic_regression.py  # Logistic Regression
    ├── adaboost.py             # AdaBoost
    └── gradient_boosting.py    # Gradient Boosting
```

---

## Architecture Overview

The project implements **Separation of Concerns** across four independent layers:

```
Controller (main.py)
    │
    ├── preprocess.py     ← load CSVs, clean, remove rare classes
    ├── embeddings.py     ← TF-IDF vectors (5000 features, bigrams)
    ├── utils.py          ← build chained label columns
    │
    ├── modelling/data_model.py   ← Data object (train/test split)
    │       target_col = chain_y2 | chain_y2_y3 | chain_y2_y3_y4
    │
    └── modelling/modelling.py    ← runs all models, prints results
            │
            ├── RandomForest
            ├── SVM
            ├── LogisticRegression
            ├── AdaBoost
            └── GradientBoosting  (all extend BaseModel)
```

### Chain Levels

| Level | Target column      | Example label                                    |
|-------|--------------------|--------------------------------------------------|
| 1     | `chain_y2`         | `Suggestion`                                     |
| 2     | `chain_y2_y3`      | `Suggestion + Payment`                           |
| 3     | `chain_y2_y3_y4`   | `Suggestion + Payment + Subscription cancellation` |

Accuracy decreases with each level because a correct prediction at Level N+1 requires Level N to also be correct.

---

## Installation

**Requirements:** Python 3.8+

Install dependencies:

```bash
pip install scikit-learn pandas numpy
```

Optional (for real translation instead of the stub):

```bash
pip install deep-translator
```

---

## How to Run

```bash
# 1. Navigate into the project folder
cd skeleton-initial

# 2. Run the pipeline
python main.py
```

The pipeline will:
1. Load and clean both CSV files from `data/`
2. Generate TF-IDF embeddings
3. Run all 5 models across all 3 chain levels
4. Print per-model classification reports
5. Print a final comparison table showing F1 and accuracy per model per level

### Expected output (summary table)

```
  MODEL ACCURACY COMPARISON (weighted F1 / Accuracy)
Model                   Chain Level 1: Type2    Chain Level 2: Type2+3  Chain Level 3: Type2+3+4
Random Forest           F1=0.70 Acc=0.71        F1=0.68 Acc=0.71        F1=0.62 Acc=0.67
SVM                     F1=0.88 Acc=0.88        F1=0.70 Acc=0.71        F1=0.63 Acc=0.64
Logistic Regression     F1=0.86 Acc=0.86        F1=0.60 Acc=0.60        F1=0.33 Acc=0.31
AdaBoost                F1=0.71 Acc=0.71        F1=0.43 Acc=0.48        F1=0.29 Acc=0.43
Gradient Boosting       F1=0.88 Acc=0.88        F1=0.65 Acc=0.69        F1=0.59 Acc=0.60

  >> Best model per level (by weighted F1):
     Chain Level 1: Type2               -> SVM  (F1=0.88, Acc=0.88)
     Chain Level 2: Type2 + Type3       -> SVM  (F1=0.70, Acc=0.71)
     Chain Level 3: Type2 + Type3 + T4  -> SVM  (F1=0.63, Acc=0.64)
```

---

## Models

All models inherit from `BaseModel` (abstract base class) and implement:

| Method           | Description                                      |
|------------------|--------------------------------------------------|
| `train(data)`    | Fit the model on `data.X_train` / `data.y_train` |
| `predict(X_test)`| Run inference and store in `self.predictions`    |
| `print_results(data)` | Print sklearn classification report        |
| `data_transform()` | Optional feature engineering hook (no-op by default) |

| Model               | Class              | Algorithm                          |
|---------------------|--------------------|------------------------------------|
| Random Forest       | `RandomForest`     | `RandomForestClassifier` (500 trees, balanced) |
| SVM ⭐ best          | `SVM`              | `LinearSVC` (balanced, max_iter=2000) |
| Logistic Regression | `LogisticReg`      | `LogisticRegression` (lbfgs, balanced) |
| AdaBoost            | `AdaBoost`         | `AdaBoostClassifier` (200 estimators) |
| Gradient Boosting   | `GradientBoosting` | `GradientBoostingClassifier` (200 estimators) |

---

## Results

SVM achieves the best weighted F1 score at every chain level. All models degrade in accuracy as the chain deepens — this is expected behaviour, since correctly predicting the Level 3 combined label requires all three individual labels to be correct simultaneously.

| Chain Level | Best Model | F1   | Accuracy |
|-------------|------------|------|----------|
| Level 1 (Type 2 only)        | SVM | 0.88 | 0.88 |
| Level 2 (Type 2 + Type 3)    | SVM | 0.70 | 0.71 |
| Level 3 (Type 2 + 3 + 4)     | SVM | 0.63 | 0.64 |

---

## How to Add a New Model

1. Create a new file in `model/`, e.g. `model/my_model.py`
2. Inherit from `BaseModel` and implement the four required methods:

```python
from model.base import BaseModel
from sklearn.naive_bayes import MultinomialNB

class NaiveBayes(BaseModel):
    def __init__(self, model_name, embeddings, y):
        super().__init__()
        self.model_name  = model_name
        self.embeddings  = embeddings
        self.y           = y
        self.predictions = None
        self.mdl         = MultinomialNB()
        self.data_transform()

    def train(self, data):
        self.mdl.fit(data.get_X_train(), data.get_type_y_train())

    def predict(self, X_test):
        self.predictions = self.mdl.predict(X_test)

    def print_results(self, data):
        from sklearn.metrics import classification_report
        print(classification_report(data.get_type_y_test(), self.predictions, zero_division=0))

    def data_transform(self):
        pass
```

3. Register it in `modelling/modelling.py`:

```python
from model.my_model import NaiveBayes

MODEL_REGISTRY = {
    ...
    'Naive Bayes': NaiveBayes,
}
```

That's it — it will be automatically included in all chain levels and the comparison table.

---

## Design Principles

| Principle | Implementation |
|-----------|---------------|
| **Separation of Concerns** | Preprocessing, embeddings, data modelling, and ML models are each in their own file/module |
| **Encapsulation** | `Data` class wraps `X_train`, `X_test`, `y_train`, `y_test` — models never access raw data directly |
| **Abstraction** | `BaseModel` (ABC) enforces a uniform interface — the controller calls the same methods regardless of which model is used |
| **Open/Closed** | Add a new model by creating one file and one registry entry — no existing code changes |
| **Configuration** | All shared constants (column names, split ratio, chain column names) live in `Config.py` |
