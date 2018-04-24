#使用Word2Vec方法,需要喂语料，corpus这里就是我们的预料, 他需要list of list 格式
import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from datetime import date

#需要学习的：
#numpy.ndarray.flatten https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
#numpy flatten的用法：https://blog.csdn.net/qq_18433441/article/details/54916991
#pandas 时间序列分析：https://www.cnblogs.com/foley/p/5582358.html

#1) 设置工作路径,划分数据集
mypath = 'F:\Downloads\lecture02'
os.chdir(mypath)
data = pd.read_csv('Combined_News_DJIA.csv')
data.head()
#划分训练集与测试集
train = data[data['Date'] < '2015-01-01']
test = data[data['Date'] < '2014-12-31']

#选取从第二列到最后一列为X_train,
X_train = train[train.columns[2:]]
#自己定义语料库：将选中的每条新闻 -> 单独的句子,集合在一起，长度为1611*25=40275个句子, 用flatten()把多维数组变为一维，默认按行
corpus = X_train.values.flatten().astype(str)

#【划分X_train】取出训练集, 一共1611行，每行用[xx,xx,xx，..] 25条XX表示里面的新闻内容，array格式
X_train = X_train.values.astype(str)
#对每一行，把25条内容合并为一条, X_train现在每行是由25条新闻组成的一个句子
X_train = np.array([' '.join(x) for x in X_train])
#【划分X_test】
X_test = test[test.columns[2:]]
X_test = X_test.values.astype(str)
X_test = np.array([' '.join(x) for x in X_test])
#【划分y_train】
y_train = train['Label'].values
#【划分y_test】
y_test = test['Label'].values

#把语料corpus（40275句新闻）打印前两个出来看看, 打印前2个X_train出来看看
#corpus[:2]
#X_train[:2]

#2）分词操作： 把句子分词nltk.tokenize， 现在语料库都是单词，X_train中现在25条新闻组成的句子的分词，
from nltk.tokenize import word_tokenize
corpus = [word_tokenize(x) for x in corpus]
X_train = [word_tokenize(x) for x in X_train]
X_test = [word_tokenize(x) for x in X_test]

#打印出来看下
#X_train[:1]
#len(X_train[:1])

# 3） 预处理，打包为一个函数
# •小写化
# •删除停止词
# •删除数字与符号
# •lemma

# 去除停用词
from nltk.corpus import stopwords

stop = stopwords.words('english')
# 去除数字，正则表达式
import re


def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))


# 特殊符号
def isSymbol(inputString):
    return bool(re.match(r'[^\w]', inputString))


# lemma提取词干
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()


# 定义check函数, 检查每一行中的每个单词是否ok
def check(word):
    if word in stop:
        return False
    elif hasNumbers(word) or isSymbol(word):
        return False
    else:
        return True


# 定义预处理函数preprocessing
def preprocessing(sen):
    res = []
    for word in sen:  # 对每一个整体的array数组
        if check(word):  # 对每一行中的每一个单词，如果check(word)返回True
            # 最小化单词，同时去除一些多余的干扰标识
            word = word.lower().replace("b'", '').replace('b"', '').replace('"', '').replace("'", '')
            # 进一步提取词干，利用append加入新的数组
            res.append(wordnet_lemmatizer.lemmatize(word))
    return res

# 处理已有的语料，X_train, X_test
corpus = [preprocessing(x) for x in corpus]
X_train = [preprocessing(x) for x in X_train]
X_test = [preprocessing(x) for x in X_test]

#打印出来看看，处理好的数据，格式就是list of list,
print(X_train[0]) #注意这里的不同，而X_train中是把25条新闻组合在一起后再进行分词（基于每天25条新闻分割）
print(corpus[0])  #corpus是以每条新闻计数的，然后在打散(基于每条新闻分割)

#4)训练NLP模型
#目前我们获得了干净的数据集合,有了人造的语料库，
#先采用Word2Vec，【此处需要了解下参数内容】，相当于转化为词向量

from gensim.models.word2vec import Word2Vec
model = Word2Vec(corpus, size=128, window=5, min_count=5, workers=4)
#打印出来试试, 注意格式问题
model.wv['victory']

#5)训练NLP模型完成后，用NLP模型表达我们的X
#我们的vec是基于每个单词的，怎么办呢？由于我们文本本身的量很小，我们可以把所有的单词的vector拿过来取个平均值：
#定义一个函数，输入一个任意word_list,得到他们的平均vector值的方法

#先获取所有的vocabulary,注意此处Genism移除了之前的model.vocab,需要替换为model.wv.voccab
vocab = model.wv.vocab
#得到任意text的vector
def get_vector(word_list):
    #新建全为0的array
    res = np.zeros([128])
    count=0
    for word in word_list:
        res += model[word]
        count += 1
    return res/count

#这样，我们可以同步把我们的X都给转化成128维的一个vector list
get_vector(['hello', 'from', 'the', 'other', 'side'])

wordlist_train = X_train
wordlist_test = X_test

#KeyError: "word 'impeached' not in vocabulary" 出现错误，这里转换失败
X_train = [get_vector(x) for x in X_train]
X_test = [get_vector(x) for x in X_test]

print(X_train[10])

#6) 建立ML模型
#这里，因为我们128维的每一个值都是连续关系的。不是分裂开考虑的。所以，道理上讲，我们是不太适合用RandomForest这类把每个column当做单独的variable来看的方法。（当然，事实是，你也可以这么用）
#我们来看看比较适合连续函数的方法：SVM

from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score

params = [0.1,0.5,1,3,5,7,10,12,16,20,25,30,35,40]
test_scores = []
for param in params:
    clf = SVR(gamma = param)
    test_score = cross_val_score(clf, X_train, y_train, cv=3, scoring='roc_auc')
    test_scores.append(np.mean(test_score))

#画图结果
import matplotlib.pyplot as plt
%matplotlib inline
plt.plot(params, test_scores)
plt.title("Param vs CV AUC Score");

#后期使用CNN方法
