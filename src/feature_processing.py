import pandas as pd
import numpy as np
import utils
import config
# -----------------
# 特征工程函数
# 根据具体内容可以进行扩展
# -----------------
def feature_engineering(df):
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    df, cat_cols = utils.one_hot_encoder(df, config.NAN_AS_CAtEGORY)
    return df