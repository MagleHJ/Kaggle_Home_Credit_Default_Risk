# Kaggle_Home_Credit_Default_Risk

## 描述

Many people struggle to get loans due to insufficient or non-existent credit histories. And, unfortunately, this population is often taken advantage of by untrustworthy lenders.

![Home Credit Group](https://storage.googleapis.com/kaggle-media/competitions/home-credit/about-us-home-credit.jpg)

[Home Credit](http://www.homecredit.net/) strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.

## 数据

![Data](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

## 工作进展

- [ ] 构建自动化测试框架
- [ ] 探索性分析
- [ ] 特征工程

## 提交记录

### baseline.csv

简单处理，具体如下：

- 使用application表
- 对Object型变量做One-hot处理
- DAYS_EMPLOYED字段缺失值修改为np.NAN（原数据中为365,243）
- 使用lightGBM分类器
- 5折正交化测试

结果

```
Local Full AUC score 0.761810
Kaggle Public Score 0.751
```

