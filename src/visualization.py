import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src import config


def plot_feature_importance(model_pipeline, model_type):
    """
    Extracts feature names and importance scores from the pipeline and plots them.
    """
    # 1. Access the preprocessor and classifier steps
    inner_pipeline = model_pipeline.named_steps['preprocessor']
    column_transformer = inner_pipeline.named_steps['preprocessor']
    classifier = model_pipeline.named_steps['classifier']

    # 2. Get feature names after transformation
    # Numerical features (first transformer in our ColumnTransformer)
    num_features = column_transformer.transformers_[0][2]

    # Categorical features (second transformer - need to get names from OneHotEncoder)
    cat_pipeline = column_transformer.transformers_[1][1]
    ohe = cat_pipeline.named_steps['onehot']
    cat_features = list(ohe.get_feature_names_out(column_transformer.transformers_[1][2]))

    all_features = num_features + cat_features

    # 3. Get importance values based on model type
    if model_type == 'logistic':
        # Use absolute values of coefficients for Logistic Regression
        importances = np.abs(classifier.coef_[0])
    else:
        # Use feature_importances_ for RF and XGBoost
        importances = classifier.feature_importances_

    # 4. Create DataFrame for plotting
    feat_imp_df = pd.DataFrame({
        'Feature': all_features,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # 5. Visualization setup
    plt.figure(figsize=(10, 8))
    sns.barplot(
        x='Importance',
        y='Feature',
        data=feat_imp_df,
        hue='Feature',
        palette='viridis',
        legend=False
    )
    plt.title(f'Feature Importance: {model_type.upper()}', fontsize=14)
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()

    # Save plot to the plots directory
    plot_dir = config.BASE_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)
    save_path = plot_dir / f"feature_importance_{model_type}.png"

    plt.savefig(save_path)
    plt.show()

