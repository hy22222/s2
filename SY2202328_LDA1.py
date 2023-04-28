from gensim.models import CoherenceModel
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import math
import jieba
import re
import os  # 用于处理文件路径
import random
import numpy as np
import multiprocessing
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# 读stopwords
f = open('cn_stopwords.txt','r',encoding='utf-8')
stopwords = f.read().splitlines()
xx = '本书来自www.cr173.com免费txt小说下载站'
xx2 = '更多更新免费电子书请关注www.cr173.com'
# 去除符号及中文无意义stopwords
def filter(text):
    a = re.sub(xx,'', text)
    b = re.sub(xx2,'', a)
    pattern = '|'.join(stopwords)
    c = re.sub(pattern,'', b)
    d = re.sub(r'\*','', c)
    e = re.sub(r'\.','',d)
    f = re.sub(r'\n','', e)
    g = re.sub(r'\u3000','', f)
    h = re.sub('\s+','',g).strip()
    return h
def fenci(con):
    char = []
    for word in con:
        char.append(word)
    con_list = char
    return con_list
def read_novel(path,j):  # 读取语料内容
    content = []
    names = os.listdir(path)
    for name in names:
            con_temp = []
            novel_name = path + '\\' + name
            with open(novel_name, 'r', encoding='ANSI') as f:
                con = f.read()
                con = filter(con)
                # con_list = fenci(con)
                con = jieba.lcut(con)  # 结巴分词
                con_list = list(con)
                pos = int(len(con)//13) ####16篇文章，分词后，每篇均匀选取13个500词段落进行建模
                for i in range(13):
                    if j == 0:
                        con_temp = con_temp + con_list[i*pos:i*pos+500]
                    else:
                        con_temp = con_temp + con_list[i*pos+501:i*pos+1000]
                content.append(con_temp)
            f.close()
    return content, names

[traindata_txt, trainfiles] = read_novel("金庸小说集",0)
train_docs =traindata_txt
labels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
[testdata_txt, testfiles] = read_novel("金庸小说集",1)
test_docs =testdata_txt
testlabels = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]

# 创建词袋表示法
dictionary = Dictionary(train_docs)
#构建语料库
train_corpus = [dictionary.doc2bow(doc) for doc in train_docs]
test_corpus = [dictionary.doc2bow(doc) for doc in test_docs]
# 训练LDA模型
num_topics =16
lda_model = LdaModel(corpus=train_corpus, id2word=dictionary,
                     num_topics=num_topics, iterations=800, passes=3, random_state=42)
train_topic_distributions = []
for doc in train_corpus:
    doc_topics = lda_model.get_document_topics(doc)
    doc_topic_distribution = np.zeros(num_topics)
    for topic_id, prob in doc_topics:
        doc_topic_distribution[topic_id] = prob
    train_topic_distributions.append(doc_topic_distribution)

# 使用得到的主题分布对测试数据进行分类
test_topic_distributions = []
for doc in test_corpus:
    doc_topics = lda_model.get_document_topics(doc)
    doc_topic_distribution = np.zeros(num_topics)
    for topic_id, prob in doc_topics:
        doc_topic_distribution[topic_id] = prob
    test_topic_distributions.append(doc_topic_distribution)

# 使用SVM对主题分布进行分类
svm = SVC(kernel='linear', C=1.0)
svm.fit(train_topic_distributions, labels)
test_predicted_labels = svm.predict(test_topic_distributions)

accuracy = accuracy_score(testlabels, test_predicted_labels)
print("Number of LDA topics: ", num_topics, ", Accuracy: ", accuracy)
print(classification_report(testlabels, test_predicted_labels))
train_predicted_labels = svm.predict(train_topic_distributions)
print(classification_report(labels, train_predicted_labels))

#不同主题数的分析
if __name__ == '__main__':
    per = []
    coh = []
    for num_topics in range(1, 17):
        lda_model = LdaModel(corpus=train_corpus, id2word=dictionary,
                             num_topics=num_topics,iterations=60, passes=10,random_state=42)
        test_topic_distributions = list(lda_model.get_document_topics(test_corpus))
        perplexity = lda_model.log_perplexity(train_corpus)
        per.append(perplexity)
        print("Perplexity: ",num_topics, perplexity)
        # 计算Coherence
        multiprocessing.freeze_support()
        coherence_model = CoherenceModel(model=lda_model, texts=train_docs, dictionary=dictionary, coherence='c_v')
        coherence = coherence_model.get_coherence()
        coh.append(coherence)
        print("Coherence: ", coherence)
    num_top=range(1,17)
    plt.plot(num_top, per,  linestyle='-', marker='.', color='r', linewidth=1,label='Perplexity')
    plt.plot(num_top, coh,  linestyle='-', marker='.', color='b', linewidth=1,label='Coherence')
    plt.show()

