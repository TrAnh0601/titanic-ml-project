from pathlib import Path

# --- Directory Structure ---
# BASE_DIR points to the root of the titanic_project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths (Following your 'raw' data convention)
DATA_RAW_DIR = BASE_DIR / 'data'
TRAIN_PATH = DATA_RAW_DIR / 'train.csv'
TEST_PATH = DATA_RAW_DIR / 'test.csv'

# Model storage
MODEL_DIR = BASE_DIR / 'models'
MODEL_DIR.mkdir(exist_ok=True)

# Submission paths
SUBMISSION_DIR = BASE_DIR / 'submission'
SUBMISSION_DIR.mkdir(exist_ok=True)
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"

# --- Titanic Specific Configurations ---
TARGET = 'Survived'

# Base features from the dataset
NUMERICAL_FEATURES = ["Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]

# --- Training Hyperparameters ---
RANDOM_STATE = 42
TEST_SIZE = 0.2