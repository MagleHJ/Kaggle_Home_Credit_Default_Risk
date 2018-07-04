from contextlib import contextmanager
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
@contextmanager
def timer(msg):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(msg, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True, categorical_columns = None):
    original_columns = list(df.columns)
    if categorical_columns == None:
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
   
# Display/plot feature importance
def display_importances(feature_importance_df_, path):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(path)