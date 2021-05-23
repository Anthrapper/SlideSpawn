import numpy as np
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from math import log
from collections import deque
from util import cosine, comm, getXnY

testno = 2
t2 = open('../Dataset/'+str(testno)+'/'+str(testno)+'.txt', 'r')
t1 = open('../Dataset/'+str(testno)+'/'+str(testno)+'p.txt', 'r')
t3 = open('../Dataset/'+str(testno)+'/title.txt', 'r')

ppt = t2.read()
paper = t1.read()
title = t3.read()


ps = PorterStemmer() 
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer_regex = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))

a=tokenizer.tokenize(paper)
b=[tokenizer_regex.tokenize(i) for i in a]
sentences=[' '.join(i) for i in b]
rm_sentences = [[j for j in sentence.lower().split() if j not in stop] for sentence in sentences]
stem_sentences = [[ps.stem(w) for w in j] for j in rm_sentences]

for i in a:
    print(i)

enumerated_sentences=list(enumerate(stem_sentences))
tf=[{} for i in range(len(stem_sentences))]

enumerated_sentences1=list(enumerate(stem_sentences1))
tf1=[{} for i in range(len(stem_sentences1))]

enumerated_sentences2=list(enumerate(stem_sentences2))
tf2=[{} for i in range(len(stem_sentences2))]

for i in enumerated_sentences:
    for j in i[1]:
        tf[i[0]][j]=tf[i[0]].get(j,0)+1
    for j in tf[i[0]]:
        tf[i[0]][j]/=len(i[1])


for i in enumerated_sentences1:
    for j in i[1]:
        tf1[i[0]][j]=tf1[i[0]].get(j,0)+1
    for j in tf1[i[0]]:
        tf1[i[0]][j]/=len(i[1])


for i in enumerated_sentences2:
    for j in i[1]:
        tf2[i[0]][j]=tf2[i[0]].get(j,0)+1
    for j in tf2[i[0]]:
        tf2[i[0]][j]/=len(i[1])


tf2

d = {}
for i in tf:
    for j in i:
        d[j] = 1
for i in tf1:
    for j in i:
        d[j] = 1
for i in tf2:
    for j in i:
        d[j] = 1


