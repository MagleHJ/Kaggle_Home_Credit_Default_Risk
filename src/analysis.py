import matplotlib.pyplot as plt
import seaborn as sns
import utils
import pandas as pd
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def generate_report(dataset, filename, pic_perfix, numerical_features, category_features):
    with utils.timer("Single variable explorer"):
        with open(filename,'w') as file:
            file.write('# 单变量分析报告\n')
        # 数值型变量
        sns.clustermap(dataset[numerical_features].corr(), center=0, cmap="vlag", linewidths=.75, figsize=(13, 13))
        num_corr_pic_path = "%s/%s.png" % (pic_perfix, 'NUMERICAL_CORR')
        plt.savefig(num_corr_pic_path)
        with open(filename,'a') as file:
            file.write('## 数值型变量间相关性\n')
            file.write("![%s](%s)\n" % ("NUMERICAL_CORR", num_corr_pic_path))
            
        for feature in numerical_features:
            filepath = '%s/%s.png' % (pic_perfix, feature)
            numerical_analysis(dataset, feature, filename, filepath)
        
        # 分类型变量
        for feature in category_features:
            filepath = '%s/%s.png' % (pic_perfix, feature)
            category_analysis(dataset, feature, filename, filepath)

def category_analysis(dataset, feature, filename, picpath, label = 'TARGET'):
    with utils.timer("Process %s" % feature):
        dataset[feature] = dataset[feature].astype('object')
        info = dataset[feature].describe()
        N = len(dataset)
        category_visual(dataset, feature, picpath, label)
        with open(filename,'a') as file:
            file.write("## %s(分类型变量)\n" % feature)
            file.write("### 统计量\n")
            file.write("有%d个可选值，它们分别是：\n" % info['unique'])
            flag = False
            for idx, value in enumerate(dataset[feature].unique()):
                file.write("* %s : %d\n" % (value, len(dataset[dataset[feature] == value])))
                if idx >=20:
                    flag = True
                    break
            if flag:
                file.write('* ...\n')

            file.write('\n')
            file.write("缺失数量：%d(%.2f%%)\n" % (N - info['count'], (N - info['count'])/N))
            file.write("频率最大的是：%s\n" % info['top'])
            file.write("### 频率图\n")
            file.write("![%s](%s)\n" % (feature, picpath))

def numerical_analysis(dataset, feature, filename, picpath, label = 'TARGET'):
    with utils.timer("Process %s" % feature):
        info = pd.concat([dataset[feature].describe(), dataset[feature].agg(['median', 'mad', 'skew', 'kurt'])],axis=0)
        N = len(dataset)
        numerical_visual(dataset, feature, picpath, label)
        corr = dataset[feature].corr(dataset[label])
        with open(filename,'a') as file:
            file.write("## %s(数值型变量)\n" % feature)
            file.write("### 统计量\n")
            file.write("取值范围：%.2f ~ %.2f\n" % (info['min'], info['max']))
            file.write("缺失数量：%d(%.2f%%)\n" % (N - info['count'], (N - info['count'])/N))
            file.write("均值：%.2f\t\t中位数：%.2f\n" % (info['mean'], info['median']))
            file.write("标准差：%.2f\t\t绝对离差：%.2f\n" % (info['std'], info['mad']))
            file.write("偏度：%.2f\t\t峰度：%.2f\n" % (info['skew'], info['kurt']))
            file.write("与预测变量的相关性：%f\n" % corr)
            file.write("|min|25%|50%|75%|max|\n")
            file.write("|-:|-:|-:|-:|-:|\n")
            file.write("|%.2f|%.2f|%.2f|%.2f|%.2f|\n" %(info['min'],
                                                        info['25%'],
                                                        info['50%'],
                                                        info['75%'],
                                                        info['max'],
                                                       ))
            file.write("### 概率密度分布\n")
            file.write("![%s](%s)\n" % (feature, picpath))

def numerical_visual(dataset, feature, filename, label='TARGET'):
    df_group = {x:y for x,y in dataset.groupby(label)}
    
    plt.figure(figsize=(8, 4))
    
    # 绘制特征的概率密度分布
    for target, df in df_group.items():
        s = sns.kdeplot(df[feature].dropna(), label="%s=%s" % (label, str(target)))
    a = plt.axes([.6, .7, .2, .2])
    s = sns.distplot(dataset[feature].dropna(), kde=True)
    
    plt.tight_layout()
    plt.savefig(filename)


def category_visual(dataset, feature, filename, label='TARGET'):
    x = []
    y = []
    y_rate = []
    for target, rcd in dataset.groupby(feature):
        x.append(target)
        y.append(len(rcd[rcd[label]==1]))
        y_rate.append(len(rcd[rcd[label]==1])/len(rcd))
        del rcd
    plt.figure(figsize=[12, 6])

    ax = plt.subplot(121)
    s = sns.barplot(x=x, y=y)
    s.set_xticklabels(s.get_xticklabels(),rotation=45)

    for a,b in zip(list(range(len(y))),y):
        ax.text(a, b, '%.3f' % b,ha='center', va= 'bottom')

    ax = plt.subplot(122)
    plt.plot(y_rate)
    plt.xticks(range(len(x)),x,rotation=45)
    for a,b in zip(list(range(len(x))),y_rate):
        ax.text(a, b, '%.3f' % b,ha='center', va= 'bottom')

    plt.tight_layout()
    plt.savefig(filename)
    gc.collect()