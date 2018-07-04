'''
数据加载
'''
import pandas as pd
from utils import timer
import utils
import gc
import config

# -------------------
# 数据加载函数
# 自动适应文件类型
# -------------------
def get_data(file_path, num_rows=None, msg = None):
    '''Method load_data
    数据加载函数
    '''
    if msg == None:
        msg = 'load data from %s' % file_path
    with timer(msg):
        file_type = file_path.split('.')[-1]
        if file_type in ('csv',):
            df = pd.read_csv(file_path, nrows=num_rows)
    return df

# ------------------
# 数据加载
# 按具体项目定制
# ------------------
def load_data():
    nrows = 10000 if config.DEBUG else None
    # 加载数据集
    ## load application.csv
    df = get_data('../input/application_train.csv', nrows)
    test_df = get_data('../input/application_test.csv', nrows)
    print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
    df = df.append(test_df).reset_index(drop=True)
    del test_df
    gc.collect()
    ## load bureau.csv/bureau_balance.csv
    bru = get_data('../input/bureau.csv', nrows)
    brub = get_data('../input/bureau_balance.csv', nrows)

    return (df, bru, brub)