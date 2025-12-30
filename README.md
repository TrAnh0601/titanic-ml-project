# Titanic Survival Prediction – Feature Engineering & Model Evaluation

## Goal
This project aims to build a robust machine learning pipeline for predicting passenger survival on the Titanic dataset, with a strong focus on **feature engineering** and **model evaluation**, rather than relying solely on complex models.

---

## Dataset
The dataset is taken from the Kaggle Titanic competition, containing passenger information such as age, gender, ticket class, fare, and family relationships.

- Task: Binary classification  
- Target variable: `Survived`

---

## Workflow Overview
1. Data preprocessing  
2. Feature engineering  
3. Model training  
4. Model evaluation and comparison  

The entire workflow is implemented using a **modular pipeline structure** to ensure clean code, reusability, and reproducibility.

---

## Feature Engineering
Several domain-informed features were created to enrich the original dataset:

- **Title extraction**  
  Passenger titles (e.g., Mr, Mrs, Miss) were extracted from names to capture social status and demographic information.

- **Family size**  
  `SibSp` and `Parch` were combined into a single feature to represent family size.

- **IsAlone**  
  A binary feature indicating whether a passenger was traveling alone.

- **Age imputation by group**  
  Missing age values were imputed using group-based statistics instead of global averages.

Categorical features were encoded and numerical features were scaled using a preprocessing pipeline.

---

## Models and Evaluation

The following models were trained and evaluated:

- **Logistic Regression**  
  Used as a strong and interpretable baseline model.

- **Random Forest**  
  Hyperparameters were tuned using **GridSearchCV** to improve performance and reduce overfitting.

- **XGBoost**  
  Trained using default (or lightly adjusted) hyperparameters to compare performance with tree-based ensembles.

### Evaluation Strategy
- Cross-validation  
- Metrics: **Accuracy** and **F1-score**

---

## Results Summary

| Model                |  Val Accurracy |  Kaggle LB  |
|----------------------|----------------|-------------|
| Logistic Regression  |     0.8100     |      -      |
| Random Forest        |     0.8492     |   0.75598   |
| Random Forest (Tuned)|     0.8212     |   0.77511   |
| XGBoost              |     0.8268     |   0.73923   |

Random Forest showed noticeable improvement after hyperparameter tuning, while Logistic Regression served as a strong baseline. XGBoost provided competitive performance with minimal tuning.

---

## Key Takeaways
- Feature engineering played a significant role in improving model performance.
- Establishing a solid baseline model is essential before tuning more complex models.
- Cross-validation is crucial for evaluating model stability and generalization.

---

## Technologies Used
- Python  
- pandas, numpy  
- scikit-learn  
- xgboost  

---

## What I Learned
- Designing reusable preprocessing and training pipelines
- Applying feature engineering to improve model performance
- Evaluating and comparing machine learning models using cross-validation
