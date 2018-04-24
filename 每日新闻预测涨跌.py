#导入常用的package
import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import  CountVectorizer, TfidfVectorizer
from datetime import date

#1) 改变路径
path = 'F:\Downloads\lecture02'
os.chdir(path)
data = pd.read_csv('Combined_News_DJIA.csv')
print(data.head())

#2) 合并25条每日新闻，分割训练集合与测试集合
#pandas.DataFrame.filter: http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.filter.html
data["combined_news"] = data.filter(regex=("Top.*")).apply(lambda x: ''.join(str(x.values)), axis=1)
#通过时间挑选训练集，测试集
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] > '2014-12-31']

#不小心发现，这个结果还不如之前的简单版；
#造成如此的原因有几种：•数据点太少，•One-Off result；
#我们到现在都只是跑了一次而已。如果我们像前面的例子一样，用Cross Validation来玩这组数据，说不定我们会发现，分数高的clf其实是overfitted了的。

#3) 提取特征features（简单版）
#这里直接调用sklearn中的TF-IDF的包来做feature，这里没有preprocessing步骤
feature_extraction = TfidfVectorizer()
#把训练集，先fit里面的文本信息，然后在来transform转换成TIIDF的格式
X_train = feature_extraction.fit_transform(train["combined_news"].values)
#这里因为feature_extraction记住了训练的格式，转换测试集的格式
X_test = feature_extraction.transform(test["combined_news"].values)
y_train = train["Label"].values
y_test = test["Label"].values

#4) 训练模型-SVM,基于高斯核函数：sklearn.svm.SVC — scikit-learn 0.19.1 documentation
#参考：https://blog.csdn.net/gamer_gyt/article/details/51265347
clf = SVC(probability=True, kernel = 'rbf')
#把X_train, y_train丢进去, 找出模型最佳参数
clf_train = clf.fit(X_train, y_train)
print(clf_train)

#5) 预测 & 评价标准
#参考：ROC和AUC介绍以及如何计算AUC：http://alexkong.net/2013/06/introduction-to-auc-and-roc/
predictions = clf.predict_proba(X_test)
print('ROC-AUC yields' + str(roc_auc_score(y_test, predictions[:,1])))

##################################################

#【进阶版】
# 文本预处理中，我们这样直接把文本放进TF-IDF，虽然简单方便，但是还是不够严谨的。 我们可以把原文本做进一步的处理。
# 把句子中分词，全部小写，去除停用词，删除一些数字，使用nltk中的lemma取出词根
# 因为外部库，比如sklearn 只支持string输入，所以我们把调整后的list再变回string: apply(lambda x : ' '.join(x))
# 后面流程一样：fit数据 -> SVM方式跑一遍 -> 看下ROC-AUC值

X_train = train["combined_news"].str.lower().str.replace('"', '').str.replace(".", '').str.split()
X_test = test["combined_news"].str.lower().str.replace('"', '').str.replace(".", '').str.split()
print(X_test[1611]) #分词后的25个句子

#删除停止词语
from nltk.corpus import stopwords
stop = stopwords.words('english')
#删除数字
import re
def hasNumbers(inputString):
    return bool(re.search(r'\d',inputString ))
#取出词根
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
#定义一个函数把需要的词语保留下来
def check(word):
    if word in stop:
        return False
    elif hasNumbers(word):
        return False
    else:
        return True
#把这些规则运用在X_train, X_test中, 留下来的单词做lemma处理
X_train = X_train.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
X_test  = X_test.apply(lambda x: [wordnet_lemmatizer.lemmatize(item) for item in x if check(item)])
print(X_test[1611]) #处理过一波的25个词语，格式为list

#接下来导入sklearn中训练，但是格式不对，需要把X_train, X_test 转换为string格式，目前是list格式
X_train = X_train.apply(lambda x: ' '.join(x))
X_test = X_test.apply(lambda x: ' '.join(x))
print(X_test[1611]) #处理过一波的25个词语，格式为string

#准备跑一遍模型
feature_extraction = TfidfVectorizer(lowercase = False)
X_train = feature_extraction.fit_transform(X_train.values)
X_test = feature_extraction.transform(X_test.values)

clf = SVC(probability=True, kernel = 'rbf')
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)
print('ROC-AUC yields' + str(roc_auc_score(y_test, predictions[:, 1])))



