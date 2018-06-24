from contextlib import contextmanager
import time
import pandas as pd

@contextmanager
def timer(msg):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(msg, time.time() - t0))

def one_hot_encoder(df, nan_as_category = True, categorical_columns = None):
    if categorical_columns == None:
        original_columns = list(df.columns)
        categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns