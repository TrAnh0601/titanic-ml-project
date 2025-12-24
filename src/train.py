import argparse
import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

from src import config
from src.dataloader import load_data
from src.pipeline import get_preprocessing_pipeline
from src.visualization import plot_feature_importance

def train(model_type, tune=False):
    # 1. Load data
    train_df, _ = load_data()

    X = train_df.drop(config.TARGET, axis=1)
    y = train_df[config.TARGET]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    # 2. Setup Pipeline
    pipeline = get_preprocessing_pipeline()

    if model_type == "logistic":
        model = LogisticRegression(random_state=config.RANDOM_STATE)
    elif model_type == "rf":
        model = RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    elif model_type == "xgboost":
        model = XGBClassifier(eval_metric='logloss', random_state=config.RANDOM_STATE)
    else:
        raise ValueError(f"Model {model_type} not supported!")

    full_pipeline = Pipeline(steps=[
        ("preprocessor", pipeline),
        ("classifier", model)
    ])

    # 3. Training
    # Tune for rf model
    if tune and model_type == "rf":
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__max_depth': [None, 5, 10, 15],
            'classifier__min_samples_split': [2, 5, 10],
            'classifier__min_samples_leaf': [1, 2, 4]
        }
        # Using 5-fold cross-validation
        grid_search = GridSearchCV(full_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)

        print(f">>> Best CV Score (Mean of 10 folds): {grid_search.best_score_:.4f}")
        print(f">>> Best Parameters: {grid_search.best_params_}")
        full_pipeline = grid_search.best_estimator_
    else:
        full_pipeline.fit(X_train, y_train)

    # 4. Evaluation
    val_acc = accuracy_score(y_val, full_pipeline.predict(X_val))
    print(f">>> Validation Accuracy: {val_acc:.4f}")

    # 5. Save model artifact
    joblib.dump(full_pipeline, config.MODEL_DIR / f"{model_type}_model.pkl")

    return full_pipeline

def generate_submission(model_pipeline, test_df):
    # Predict results
    predictions = model_pipeline.predict(test_df)

    # Create submission DataFrame according to Kaggle format
    submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": predictions
    })

    # Save to CSV
    submission.to_csv(config.SUBMISSION_PATH, index=False)

if __name__ == "__main__":
    _, test_df = load_data()
    parser = argparse.ArgumentParser(description="Titanic Model Training CLI")
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["logistic", "rf", "xgboost"],
    )
    parser.add_argument("--tune", action="store_true")

    args = parser.parse_args()
    fitted_model = train(args.model, tune=args.tune)
    plot_feature_importance(fitted_model, args.model)
    generate_submission(fitted_model, test_df)





