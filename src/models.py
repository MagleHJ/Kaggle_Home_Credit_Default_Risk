import pandas as pd
import numpy as np
import config
import gc
import utils
from sklearn.metrics import roc_auc_score, roc_curve

def get_model(df, feats):
    res,feature_importance_df  = lightgbm(df, feats)
    res.to_csv(config.SUBMISSION_FILE_NAME, index= False)
    utils.display_importances(feature_importance_df, config.IMPORTANCE_IMAGE_PATH)

# ------------------
# 模型
# ------------------


from lightgbm import LGBMClassifier


def lightgbm(df, feats):
    train_df = df[df[config.TARGET].notnull()]
    test_df = df[df[config.TARGET].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    
    # 创建存放结果的数组
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()

    for n_fold, train_idx, train_x, train_y, valid_idx, valid_x, valid_y in kflod(train_df, feats):
        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1, )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 100)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / config.NUM_FLODS

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    print('Full AUC score %.6f' % roc_auc_score(train_df[config.TARGET], oof_preds))
    # Write submission file and plot feature importance
    test_df[config.TARGET] = sub_preds
    return test_df[[config.RECORD_ID_FEILD, config.TARGET]], feature_importance_df

# ---------------------
# 相关工具函数
# ---------------------

# K-Fold


from sklearn.model_selection import KFold, StratifiedKFold


def kflod(df, feats):
    target = config.TARGET
    if config.STRATIFIED:
        flods = StratifiedKFold(n_splits=config.NUM_FLODS, shuffle=True, random_state=1001)
    else:
        flods = KFold(n_splits=config.NUM_FLODS, shuffle=True, random_state=1001)
    for n_fold, (train_idx, valid_idx) in enumerate(flods.split(df[feats], df[target])):
        train_x, train_y = df[feats].iloc[train_idx], df[target].iloc[train_idx]
        valid_x, valid_y = df[feats].iloc[valid_idx], df[target].iloc[valid_idx]
        yield n_fold, train_idx, train_x, train_y, valid_idx, valid_x, valid_y
