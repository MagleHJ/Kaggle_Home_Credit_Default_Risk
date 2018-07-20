import pandas as pd
import numpy as np
import utils
import config
import gc
from pomegranate import BayesianNetwork

# -----------------
# 特征工程函数
# 根据具体内容可以进行扩展
# -----------------


def feature_engineering(data_list):
    df, bru, brub, prev, pos, ins, cc = data_list
    del data_list
    gc.collect()

    with utils.timer('Process application'):
        df = get_feature_from_app(df)

    with utils.timer('Process bureau & balance'):
        bru_agg = get_feature_from_bru(bru, brub)
        df = df.join(bru_agg, how='left', on='SK_ID_CURR')
    
    with utils.timer('Process previous application'):
        prev_agg = get_feature_from_pre_app(prev)
        df = df.join(prev_agg, how='left', on='SK_ID_CURR')

    with utils.timer('Process POS CASH balance'):
        pos_agg = get_feature_from_pos_cash(pos)
        df = df.join(pos_agg, how='left', on='SK_ID_CURR')

    with utils.timer('Process installment payments'):
        ins_agg = get_feature_from_inst_pay(ins)
        df = df.join(ins_agg, how='left', on='SK_ID_CURR')

    with utils.timer('Process credit card balance'):
        cc_agg = get_feature_from_credit_card_balance(cc)
        df = df.join(cc_agg, how='left', on='SK_ID_CURR')
    
    return df

def get_feature_from_app(df):

    # 合并
    df['WEEKDAY_APPR_PROCESS_START_MAPPED'] = df['WEEKDAY_APPR_PROCESS_START'].map({'SUNDAY':'1','SATURDAY':'1','MONDAY':'1',
                                                                              'THURSDAY':'2','FRIDAY':'2','WEDNESDAY':'2',
                                                                              'TUESDAY':'3'})
    df['HOUR_APPR_PROCESS_START_MAPPED'] = df['HOUR_APPR_PROCESS_START'].apply(lambda hour:'worktime' if(hour>=9 and hour<=21) else 'sleeptime')
    df['NAME_TYPE_SUITE_MAPPED'] = df['NAME_TYPE_SUITE'].map({'Family':'Family','Children':'Family',
                                                                                    'Unaccompanied':'Partner','Spouse, partner':'Partner',
                                                                                    'Group of people':'Partner',
                                                                                    'Other_A':'Other','Other_B':'Other'})
    df['WALLSMATERIAL_MODE_MAPPED'] = df['WALLSMATERIAL_MODE'].replace(['Stone, brick', 'Others', 'Mixed', 'Block', 'Panel'], ['General']*5)
    df['NAME_HOUSING_TYPE_MAPPED'] = df['NAME_HOUSING_TYPE'].map({'Rented apartment':'No house','With parents':'No house',
                                                                'Office apartment':'Office',
                                                                'House / apartment':'Owner','Municipal apartment':'Owner','Co-op apartment':'Owner'})
    #NAME_EDUCATION_TYPE
    df['NAME_EDUCATION_TYPE_MAPPED'] = df['NAME_EDUCATION_TYPE'].replace({'Academic degree':'Higher education'})
    degree_mapDict = {"Lower secondary":0,"Secondary / secondary special":1,"Incomplete higher":2,"Higher education":3}
    df['NAME_EDUCATION_TYPE_MAPPED'] = df['NAME_EDUCATION_TYPE_MAPPED'].map(degree_mapDict)
    #NAME_FAMILY_STATUS
    df['NAME_FAMILY_STATUS_MAPPED'] = df['NAME_FAMILY_STATUS'].replace({'Unknown':'married'})
    #将孩子数量3个以及3个以上的归为一类
    df['CNT_CHILDREN_NAPPED'] = df['CNT_CHILDREN'].apply(lambda x:3 if x>=3 else x)
    #NAME_INCOME_TYPE
    df['NAME_INCOME_TYPE_MAPPED'] = df['NAME_INCOME_TYPE'].replace({'Student':'Unemployed','Maternity leave':'Unemployed','Businessman':'Commercial associate'})
    #NAME_INCOME_TYPE是Pensioner，Unemployed 另OCCUPATION_TYPE和ORGANIZATION_TYPE为 no job
    df.loc[df['NAME_INCOME_TYPE'].isin(['Pensioner','Unemployed']),['OCCUPATION_TYPE','ORGANIZATION_TYPE']]='no job'
    #采用BN预测填补CCUPATION_TYPE 
    miss_data=df.loc[df['OCCUPATION_TYPE'].isnull()]
    complete_data=df[df['OCCUPATION_TYPE'].notnull()]
    miss_data['OCCUPATION_TYPE'].replace({np.nan:None},inplace=True)#将NaN格式转化成None
    bn_model = BayesianNetwork.from_samples(complete_data[['NAME_INCOME_TYPE','ORGANIZATION_TYPE','OCCUPATION_TYPE']], algorithm='exact-dp')
    predict = bn_model.predict(miss_data[['NAME_INCOME_TYPE','ORGANIZATION_TYPE','OCCUPATION_TYPE']].values.tolist())
    miss_data.loc[:,['OCCUPATION_TYPE']] = [each[-1] for each in predict]
    df=pd.concat([complete_data,miss_data])#将填补后的数据集合并
    df['OCCUPATION_TYPE'].replace({'Managers':'Core staff',"Drivers":'unstable','Cleaning staff':'unstable',
                                                'Cooking staff':'unstable','Security staff':'unstable','Waiters/barmen staff':'unstable',
                                                'Secretaries':'High skill tech staff','IT staff':'High skill tech staff',
                                                'HR staff':'High skill tech staff','Realty agents':'High skill tech staff',
                                                'Medicine staff':'High skill tech staff','Private service staff':'High skill tech staff'},inplace=True)
    #工龄
    df['gongling']=0
    df.loc[df['DAYS_EMPLOYED']== 365243,'gongling']='no job'
    df.loc[df['DAYS_EMPLOYED']< -2400,'gongling']='long'
    df.loc[(df['DAYS_EMPLOYED']>= -2400) & (df['DAYS_EMPLOYED']< 1),'gongling']='short'
    #私人电话邮件提供数量
    df['FLAG_MOBIL'].replace({0:1},inplace=True)
    per_phone_cols=['FLAG_MOBIL',  'FLAG_PHONE', 'FLAG_EMAIL']
    df['NUM_PER_PHONE_PROVIDED']=df.loc[:,per_phone_cols].sum(axis=1)
    #工作相关电话提供数量
    wor_phone_cols=[  'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE']
    df['NUM_WOR_PHONE_PROVIDED']=df.loc[:,wor_phone_cols].sum(axis=1)
    phone_cols=['FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL']
    df.drop( phone_cols,axis = 1,inplace=True)

    # 提取
    df['NUM_DIFF_IN_CITY_LEVEL'] = df['REG_CITY_NOT_LIVE_CITY'] + df['REG_CITY_NOT_WORK_CITY'] + df['LIVE_CITY_NOT_WORK_CITY']
    df['NUM_DIFF_IN_REGION_LEVEL'] = df['REG_REGION_NOT_LIVE_REGION'] + df['REG_REGION_NOT_WORK_REGION'] + df['LIVE_REGION_NOT_WORK_REGION']
    df['NUM_DIFF_IN_AREA'] = df['NUM_DIFF_IN_REGION_LEVEL'] + df['NUM_DIFF_IN_CITY_LEVEL']
    document_cols_except3=['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6','FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8',
            'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_13','FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',
            'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21' ]
    df['NUM_DOCUMENT_PROVIDED_EXCEPT3']=df.loc[:,document_cols_except3].sum(axis=1)
    df['NUM_DOCUMENT_PROVIDED_EXCEPT3'].replace({3:2,4:2},inplace=True)

    # 丢弃
    mapped_feilds = []
    drop_feilds = [
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
        'CNT_FAM_MEMBERS',
    ]
    drop_feilds = drop_feilds + ['FLAG_DOCUMENT_%d' % num for num in range(2, 22) if num!=3]
    for col in df.columns:
        if '_MAPPED' in col:
            mapped_feilds.append(col[:-7])
    df = df[df.columns[~df.columns.isin(mapped_feilds+drop_feilds)]]

    df, cat_cols = utils.one_hot_encoder(df, config.NAN_AS_CAtEGORY)

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
    days_list = (0, -1, -2, -3, -6, -12, -18, -24, -36, -60, -96)
    for days in days_list:
        for status in original_columns:
            feild = 'OVERDUE_STATUS_%s_IN_PAST_%d_DAYS' % (status, abs(days))
            brub_agg[feild] = brub_agg[status] >= days
    for col in original_columns:
        del brub_agg[col]
    for col in brub_agg.columns:
        brub_agg[col] = brub_agg[col].astype(int)
    bru_agg = bru_agg.join(brub_agg, how='left', on='SK_ID_CURR')
    return bru_agg

def get_feature_from_pre_app(prev):

    prev, cat_cols = utils.one_hot_encoder(prev, nan_as_category= True) 
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

    # Add feature: value received / value ask
    prev['APP_CREDIT_PERC'] = prev['AMT_CREDIT'] / prev['AMT_APPLICATION']

    # Previous applications' numeric features
    num_aggregations = {
        'AMT_ANNUITY' : ['max', 'mean'],
        'AMT_APPLICATION': ['max', 'mean'],
        'AMT_CREDIT': ['max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'RATE_INTEREST_PRIMARY': ['max', 'mean'],
        'RATE_INTEREST_PRIVILEGED': ['max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }

    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left', on='SK_ID_CURR')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left', on='SK_ID_CURR')
    del refused, refused_agg, approved, approved_agg, prev
    gc.collect()
    return prev_agg

# Preprocess POS_CASH_balance.csv
def get_feature_from_pos_cash(pos):
    pos, cat_cols = utils.one_hot_encoder(pos, nan_as_category= True)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
    del pos
    gc.collect()
    return pos_agg
    
# Preprocess installments_payments.csv
def get_feature_from_inst_pay(ins):
    ins, cat_cols = utils.one_hot_encoder(ins, nan_as_category= True)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum'],
        'DBD': ['max', 'mean', 'sum'],
        'PAYMENT_PERC': [ 'mean',  'var'],
        'PAYMENT_DIFF': [ 'mean', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def get_feature_from_credit_card_balance(cc):
    cc, cat_cols = utils.one_hot_encoder(cc, nan_as_category= True)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg