import pickle
import numpy as np
import nltk
import nltk.data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from math import log
from collections import deque
from textblob import TextBlob
import cvxpy
from util import getXnY



clf=pickle.load(open('clf.sav','rb'))

paper = open('ppt.txt', 'r').read()
ps = PorterStemmer() 
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('large_grammars')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')




tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
tokenizer_regex = RegexpTokenizer(r'\w+')
stop = set(stopwords.words('english'))
grammar = nltk.data.load('grammars/large_grammars/atis.cfg')
parser = nltk.parse.BottomUpChartParser(grammar)



def stem(a):
    if ps.stem(a)==a:
        return a
    return stem(ps.stem(a))

def Roman(s):
    if s[0].isdigit():
        return 1
    a = ["M", "D", "C", "L", "X", "V", "I"]
    for i in s:
        if i not in a:
            return 0
    return 1

a=tokenizer.tokenize(paper)
b=[tokenizer_regex.tokenize(i) for i in a]
titl = {"Abstract":["Abstract"],"Introduction":["Introduction","Motivation"],"Related Work":["et","Literature Survey","Related Work"],"Proposed Method":["Model","Proposed","Technique"],"Results":["Experiments", "Evaluations","Results" ],"Conclusion":["Conclusion"], "References":["Reference"]}
title_sentence=[]
cur="Introduction"
p=set()
for i in range(len(b)):
    for j in titl:
        f=0
        for k in titl[j]:
            if len(b[i])>=2 and (stem(b[i][0]+' '+b[i][1])==stem(k) or (Roman(b[i][0]) and stem(b[i][1])==stem(k))):
                f=1
                break
            
            if len(b[i])>=3 and Roman(b[i][0]) and stem(b[i][1]+' '+b[i][2])==stem(k):
                f=1
                break
            if  len(b[i])>=1 and stem(b[i][0])==stem(k):
                f=1
                break
                 
        if f:
            if j=="Related work" or j=="Proposed Method":
                if "Introduction" not in p:
                    break
            elif j=="Results":
                if "Proposed Method" not in p:
                    break
            elif j=="Conclusion":
                if "Results" not in p:
                    break
            cur=j
            break
    p.add(cur)
    title_sentence.append(cur)

sentences=[' '.join(i) for i in b]



phrase=set()
c=0
m={}
for sentence in sentences:
    for j in TextBlob(sentence).noun_phrases:
        if len(j)<=2:
            continue
        phrase.add((j,c))
        m[j]=m.get(j,0)+1
    c+=1



#global phrases    
gp=[]
m_gp={}
thresh=2
for i in m:
    if m[i]>=thresh:
        gp.append(i)
for i in range(len(gp)):
    m_gp[gp[i]]=i



#local phrases
lp=[]
temp=[[] for i in sentences]
for p,i in phrase:
    if p in gp:
        temp[i].append(len(lp))
        lp.append(p+'_'+title_sentence[i])

sentences_phrase=[[0 for j in lp] for i in sentences]
for i in range(len(temp)):
    for j in temp[i]:
        sentences_phrase[i][j]=1
gp_lp=[[0 for j in lp] for i in gp]
c=0
for i in lp:
    gp_lp[m_gp[i.split('_')[0]]][c]=1
    c+=1




x=cvxpy.Variable(len(sentences),boolean=True)
lpp=cvxpy.Variable(len(lp),boolean=True)
gpp=cvxpy.Variable(len(gp),boolean=True)
y=cvxpy.Variable(len(sentences),boolean=True)
LMax=1000
c1=np.array([len(i) for i in b])*x<=LMax
c2=np.array(sentences_phrase)*lpp>=x
c3=np.array(sentences_phrase).T*x>=lpp
c4=np.array(sentences_phrase)*lpp>=y
c5=np.array(gp_lp)*lpp>=gpp
c6=np.array(gp_lp).T*gpp>=lpp
c7=np.ones(len(gp))*gpp*2<=np.ones(len(sentences))*x
X=getXnY(2,0)
w=clf.predict(X)
score=np.array([len(i)/LMax for i in b])*np.array(w)*x+np.array([i/len(w) for i in w])*y
objective = cvxpy.Problem(cvxpy.Maximize(score), [c1,c2,c3,c4,c5,c6,c7])
objective.solve(solver=cvxpy.GLPK_MI)

ppt={}
for i in range(len(x.value)):
    if x.value[i]:
        if title_sentence[i] in ppt:
            ppt[title_sentence[i]].append(sentences[i])
        else:
            ppt[title_sentence[i]]=[sentences[i]]

print(ppt)

print(title_sentence)


