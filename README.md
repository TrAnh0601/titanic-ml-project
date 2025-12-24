# Titanic: Modular Machine Learning Pipeline

A binary classification project built with a **production-ready modular architecture**.

## Key Features
* **Modular Design**: Decoupled components (`src/`) for configuration, feature engineering, and model training following the **"Simplicity is King"** principle.
* **Custom Transformers**: Implemented `TitleExtractor`, `FamilySizeAdder`, and a title-based `AgeImputer` to capture deep data insights.
* **Unified Pipeline**: End-to-end Scikit-learn `Pipeline` integrating `SimpleImputer` safety nets and `StandardScaler` to prevent data leakage.
* **Hyperparameter Tuning**: Systematic optimization using `GridSearchCV` with **10-fold Cross-Validation**.



## Results & Benchmarks

| Model | Val Accuracy | 10-Fold CV Score | Kaggle LB |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 0.8100 | - | - |
| Random Forest (Default) | **0.8492** | - | 0.75598 |
| **Random Forest (Tuned)** | 0.8212 | **0.8413** | **0.77511** |
| XGBoost | 0.8268 | - | 0.73923 |

**Insight**: The tuned model achieved a higher **CV Score (0.8413)** compared to its single-split performance, indicating superior generalization despite the lower validation score.

## Quick Start
1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train & Evaluate**:
    ```bash
    python -m src.train --model rf --tune
    ```
