import pandas as pd
import numpy as np
import utils
import config
import gc


# -----------------
# 特征工程函数
# 根据具体内容可以进行扩展
# -----------------


def feature_engineering(data_list):
    df, bru, brub = data_list
    del data_list
    gc.collect()

    with utils.timer('Process application'):
        df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
        df, cat_cols = utils.one_hot_encoder(df, config.NAN_AS_CAtEGORY)

    with utils.timer('Process bureau & balance'):
        bru_agg = get_feature_from_bru(bru, brub)
        df = df.join(bru_agg, how='left', on='SK_ID_CURR')
    return df


def get_feature_from_bru(bru, brub):

    bru = bru.set_index('SK_ID_BUREAU')

    # brub
    # 统计该笔贷款出现逾期的情况
    # 记录最近的逾期时间
    brub.loc[brub['STATUS'].isin(['C','X']), 'STATUS'] = '0'    # 将贷款结清或者状态未知当作未逾期
    overdue = brub[brub['STATUS'] != '0']

    STATUS = '12345'
    brub_cat = []
    for status in STATUS:
        feild = 'LAST_OVERDUE_IN_STATUS_%s' % status
        df = overdue[overdue['STATUS'] == status]
        df = df.groupby('SK_ID_BUREAU').agg({'MONTHS_BALANCE': 'max'})
        bru[feild] = df['MONTHS_BALANCE']
        brub_cat.append(feild)

    del brub, df, overdue
    gc.collect()
    # bru
    num_aggregations = {
        'AMT_CREDIT_MAX_OVERDUE': ['mean', 'max', 'var'],
        'DAYS_CREDIT': ['mean', 'var', 'min'],
        'DAYS_CREDIT_UPDATE': ['mean', 'max'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['mean', 'sum', 'min', 'max', 'var'],
        'AMT_CREDIT_SUM_DEBT': ['mean', 'sum', 'max'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean', 'max', 'sum', 'var'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum', 'min', 'max', 'var'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum', 'mean', 'var', 'max'],
        'DAYS_ENDDATE_FACT': ['mean', 'max'],
        'DAYS_CREDIT_ENDDATE': ['mean', 'max', 'var', 'sum'],
    }
    bru, bru_cat = utils.one_hot_encoder(bru, categorical_columns=['CREDIT_TYPE', 'CREDIT_ACTIVE'])
    cat_aggregations = {}
    for cat in bru_cat: cat_aggregations[cat] = ['mean']

    bru_agg = bru.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bru_agg.columns = pd.Index(['BRU_ORIGIN_' + e[0] + "_" + e[1].upper() for e in bru_agg.columns.tolist()])

    # Bureau: Active credits - using only numerical aggregations
    active = bru[bru['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['BRU_ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bru_agg = bru_agg.join(active_agg, how='left', on='SK_ID_CURR')
    del active, active_agg
    gc.collect()

    # Bureau: Closed credits - using only numerical aggregations
    closed = bru[bru['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['BRU_CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bru_agg = bru_agg.join(closed_agg, how='left', on='SK_ID_CURR')
    del closed, closed_agg
    gc.collect()

    # bru_brub
    brub_cat.append('SK_ID_CURR')
    brub = bru[brub_cat]
    brub_cat.remove('SK_ID_CURR')
    brub_aggregations = {}
    for cat in brub_cat: brub_aggregations[cat] = ['max']
    brub_agg = brub.groupby('SK_ID_CURR').agg(brub_aggregations)

    brub_agg.columns = pd.Index([e[0][-1] for e in brub_agg.columns.tolist()])
    original_columns = brub_agg.columns
    for days in range(-96, 1):
        for status in original_columns:
            feild = 'OVERDUE_STATUS_%s_IN_PAST_%d_DAYS' % (status, abs(days))
            brub_agg[feild] = brub_agg[status] >= days
    for col in original_columns:
        del brub_agg[col]
    for col in brub_agg.columns:
        brub_agg[col] = brub_agg[col].astype(int)
    bru_agg = bru_agg.join(brub_agg, how='left', on='SK_ID_CURR')
    return bru_agg
