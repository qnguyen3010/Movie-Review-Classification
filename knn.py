#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 18:41:33 2017

@author: AaronNguyen
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix


# read the dataset
import csv   
def loadDataSet(filename, dataSet=[]):
	with open(filename, 'rt', ) as csvfile:
	    lines = csv.reader(csvfile, delimiter = '\t')
	    dataset = list(lines)
	    for x in range(len(dataset)):
	        dataSet.append(dataset[x])

dataSet=[]
testSet=[]
loadDataSet('train.dat.txt', dataSet)
loadDataSet('test.dat.txt', testSet)
print ('Dataset: ' + repr(len(dataSet)))
print ('Dataset: ' + repr(len(testSet)))

# remove HTML tags
import re
TAG_RE = re.compile(r'<[^>]+>')
for j in range(len(dataSet)):
    dataSet[j][1] = TAG_RE.sub('',dataSet[j][1])
for i in range(len(testSet)):
    testSet[i][0] = TAG_RE.sub('',testSet[i][0])

# seperate review and sentiment scores in the dataSet and testSet
pre_names = []
cls = []
pre_test = []

for i in range(len(dataSet)):
    pre_names.append(dataSet[i][1])
for j in range(len(dataSet)):
    cls.append(dataSet[j][0])
for k in range(len(testSet)):
    pre_test.append(testSet[k][0])

# remove all stopwords
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
def removeStopWord(list):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(list)
    filtered_list = []
    for w in word_tokens:
        if w not in stop_words:
            if len(w) >= 3:
                filtered_list.append(w.lower())
    return filtered_list

names = []
for line in pre_names:
    filtered_line = removeStopWord(line)
    names.append(filtered_line)

test = []
for line in pre_test:
    filtered_line = removeStopWord(line)
    test.append(filtered_line)

# create the dictionary of words for each dataset
def build_dict(docs):
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
    return nrows,ncols,nnz,idx
      
# create the sparse matrix from a dataset
def build_matrix(docs,nrows,ncols,nnz,idx):
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            if k in idx:
                ind[j+n] = idx[k]
                val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat

# normalize the sparse matrix
def csr_l2normalize(mat, copy=False, **kargs):
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat
    
# apply kNN algorithm to find out the sentiment score for a review
def classify(x, train, clstr):
        k = 20
        # find nearest neighbors for x
        dots = x.dot(train.T)
        sims = list(zip(dots.indices, dots.data))
        if len(sims) == 0:
            # could not find any neighbors
            return '+1' if np.random.rand() > 0.5 else '-1'
        sims.sort(key=lambda x: x[1], reverse=True)
        tc = Counter(clstr[s[0]] for s in sims[:k]).most_common(2)
        if len(tc) < 2 or tc[0][1] > tc[1][1]:
            # majority vote
            return tc[0][0]
        # tie break
        tc = defaultdict(float)
        for s in sims[:k]:
            tc[clstr[s[0]]] += s[1]
        return sorted(tc.items(), key=lambda x: x[1], reverse=True)[0][0]

# classify all test set and store the result
def classifyKnn(names, cls,test, c=3, k=10, d=2):
    nrows,ncols,nnz,idx = build_dict(names)
    mat = build_matrix(names,nrows,ncols,nnz,idx)
    nnz1 = 0
    for doc in test:
        nnz1 += len(set(doc))
    test_mat = build_matrix(test,nrows,ncols,nnz1,idx)
    csr_l2normalize(mat)
    csr_l2normalize(test_mat)
    print (mat.shape)
    print (test_mat.shape)
    clspr = []
    for i in range(test_mat.shape[0]):
        clspr.append(classify(test_mat[i,:], mat, cls))
        if i%100 == 0:
            print (i)
    return clspr



result = classifyKnn(names,cls,test)
print(result)
f = open("result4.dat.txt", "w")
f.write("\n".join(map(lambda x: str(x), result)))
f.close()


