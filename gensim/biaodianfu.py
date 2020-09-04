# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 11:39:40 2020

@author: Qihang Yang
"""

import numpy as np
import matplotlib.pyplot as plt
import gensim
from gensim.utils import open
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.metrics import roc_curve, auc
import random
 

#%%

import os
import os.path
import time
time1=time.time()

##########################合并同一个文件夹下多个txt################
def MergeTxt(filepath,outfile):
    k = open(filepath+outfile, 'a+')
    for parent, dirnames, filenames in os.walk(filepath):
        for filepath in filenames: 
            print(filepath)
            txtPath = os.path.join(parent, filepath)  # txtpath就是所有文件夹的路径
            f = open(txtPath)
            ##########换行写入##################
            k.write(f.read()+"\n")
    k.close()
    return "finished"

filepath="aclImdb/train/unsup/"
outfile="unsup_train.txt"
MergeTxt(filepath,outfile)
time2 = time.time()
print '总共耗时：' + str(time2 - time1) + 's'
    
    
#%% 
 
# 读取影评内容
with open('pos_train.txt', 'r', encoding='utf-8') as infile:
    pos_reviews = []
    line = infile.readline()
    while line:
        pos_reviews.append(line)
        line = infile.readline()
 
with open('neg_train.txt', 'r', encoding='utf-8') as infile:
    neg_reviews = []
    line = infile.readline()
    while line:
        neg_reviews.append(line)
        line = infile.readline()
 
with open('unsup_train.txt', 'r', encoding='utf-8') as infile:
    unsup_reviews = []
    line = infile.readline()
    while line:
        unsup_reviews.append(line)
        line = infile.readline()
 
# 数据划分, 1代表积极情绪，0代表消极情绪
y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))
x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)
 
 
def labelize_reviews(reviews):
    for i, v in enumerate(reviews):
        yield gensim.utils.simple_preprocess(v, max_len=100)
x_train_tag = list(labelize_reviews(x_train))
x_test_tag = list(labelize_reviews(x_test))
unsup_reviews_tag = list(labelize_reviews(unsup_reviews))
 
 
model = Word2Vec(size=200, window=10, min_count=1)
# 对所有评论创建词汇表
all_data = x_train_tag
all_data.extend(x_test_tag)
all_data.extend(unsup_reviews_tag)
model.build_vocab(all_data)
 
 
def sentences_perm(sentences):
    shuffled = list(sentences)
    random.shuffle(shuffled)
    return (shuffled)
 
 
for epoch in range(10):
    print('EPOCH: {}'.format(epoch))
    model.train(sentences_perm(all_data), total_examples=model.corpus_count, epochs=1)
 
 
def build_word_vector(text, size=200):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in text:
        try:
            vec += model[word]
            count += 1.
        except KeyError:
            continue
    if count != 0:
        vec /= count
    return vec
 
 
train_vecs = np.concatenate([build_word_vector(gensim.utils.simple_preprocess(z, max_len=200)) for z in x_train])
test_vecs = np.concatenate([build_word_vector(gensim.utils.simple_preprocess(z, max_len=200)) for z in x_test])
train_vecs = scale(train_vecs)
test_vecs = scale(test_vecs)
classifier = LogisticRegression()
classifier.fit(train_vecs, y_train)
 
print(classifier.score(test_vecs, y_test))
y_prob = classifier.predict_proba(test_vecs)[:, 1]
 
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='area = %.2f' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.legend(loc='lower right')
plt.show()