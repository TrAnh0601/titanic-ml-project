from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src import config
from src.features import TitleExtractor, FamilySizeAdder, AgeImputerByTitle

def get_preprocessing_pipeline():
    """
    Create a comprehensive pipeline
    Feature Engineering -> Imputation -> Scaling/Encoding
    """

    # 1.Feature Engineering (Sequential)
    # These transformers run on the whole dataframe
    feature_engineering_pipeline = Pipeline(steps=[
        ('title_extractor', TitleExtractor()),
        ('family_size', FamilySizeAdder()),
        ('age_imputer', AgeImputerByTitle())
    ])

    # 2.Column-specific transformations
    # Define which columns get scaled and which get encoded

    # Numerical: Scale after imputation and feature engineering
    numeric_cols = config.NUMERICAL_FEATURES + ["FamilySize"]
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Encode Title, Sex, and Embarked
    categorical_cols = config.CATEGORICAL_FEATURES + ["Title"]
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'  # Drop columns not specified (like Name, Ticket, PassengerId)
    )

    # 3.Final Full Pipeline
    full_pipeline = Pipeline(steps=[
        ('features', feature_engineering_pipeline),
        ('preprocessor', preprocessor)
    ])

    return full_pipeline



