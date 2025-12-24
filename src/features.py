import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class TitleExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract titles from the 'Name' column.
    """
    def __init__(self):
        # Define mapping for title simplification
        self.title_map = {
            "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
            "Dr": "Rare", "Rev": "Rare", "Col": "Rare", "Major": "Rare",
            "Mlle": "Miss", "Mme": "Mrs", "Don": "Rare", "Lady": "Rare",
            "Countess": "Rare", "Jonkheer": "Rare", "Sir": "Rare",
            "Capt": "Rare", "Ms": "Miss"
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Extract title using regex
        X['Title'] = X['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

        # Map rare titles to a common category
        X['Title'] = X['Title'].map(self.title_map).fillna("Rare")

        return X

class FamilySizeAdder(BaseEstimator, TransformerMixin):
    """
    Combine SibSp and Parch to create FamilySize and IsAlone features.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1
        X['IsAlone'] = np.where(X['FamilySize'] == 1, 1, 0)
        return X

class AgeImputerByTitle(BaseEstimator, TransformerMixin):
    """
    Impute missing Age values based on the median age of each Title group.
    """
    def __init__(self):
        self.group_medians_ = {}

    def fit(self, X, y=None):
        # Calculate median age for each Title
        self.group_medians_ = X.groupby('Title')['Age'].median().to_dict()
        return self

    def transform(self, X):
        X = X.copy()
        # Fill missing Age using the pre-calculated medians from training data
        X['Age'] = X.apply(
            lambda row: self.group_medians_.get(row['Title'], 28) if np.isnan(row['Age']) else row['Age'],
            axis=1
        )
        return X

