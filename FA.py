import numpy as np
import pandas as pd
import math
import csv
import random
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier


def sigmoid(t, rn):
  a=np.zeros(d-1)
  for i in range(d-1):
    if rn[i] < (math.exp(2 * abs(t[i]))-1)/(math.exp(2* abs(t[i]))+1):
      a[i]=1
    else:
      a[i]=0
  return a


def init_fitness(sample):
  fitness = np.zeros([len(sample),1])
  for i in range(len(sample)):
    row = sample[i, :]
    col = []
    for j in range(41):
      if row[j] == 1.0:
        col.append(str(j))
    stripped_data = np.array(DS[col])
    # splits stripped_data into train_x and test_x , target_data into train_y and test_y in 7:3 ratio
    train_x, test_x, train_y, test_y = train_test_split(stripped_data, target_data, test_size=0.3)
    DCST = DecisionTreeClassifier(max_depth=41,criterion="gini")
    DCST.fit(train_x, train_y.ravel())
    pred_y = DCST.predict(test_x)
    cfm = confusion_matrix(test_y, pred_y)
    fitness[i,0] = np.trace(cfm) / np.sum(cfm)
  return fitness


def inter_fitness(sample):
  row = sample
  col = []
  for j in range(41):
    if row[j] == 1.0:
      col.append(str(j))
  stripped_data = np.array(DS[col])
  train_x, test_x, train_y, test_y = train_test_split(stripped_data, target_data, test_size=0.3)
  DCST = DecisionTreeClassifier(max_depth=41,criterion="gini")
  DCST.fit(train_x, train_y.ravel())
  pred_y = DCST.predict(test_x)
  cfm = confusion_matrix(test_y, pred_y)
  fitness = np.trace(cfm) / np.sum(cfm)
  return fitness


def modFA(k,rec,ppl,fit,lbest,bestitr):
  for e in range(itr):
    bestchoice = np.zeros([n,(2*d)-1])
    e_alp = pow(alp_init * (1-(e/itr)),2) * (ub-lb)
    for j in range(k,n):
      lbdis = np.zeros([d-1])
      gbdis = np.zeros([d-1])
      u = np.zeros([k])
      rl = np.linalg.norm(lbest[j] - ppl[j])
      rg = np.linalg.norm(lbest[0] - ppl[j])
      lbdis = c1 * np.exp(-gamma * pow(rl,2)) * (lbest[j] - ppl [j])
      gbdis = c2 * np.exp(-gamma * pow(rg,2)) * (lbest[0] - ppl[j])
      #print("lbdis:\n",lbdis)
      #print("gbdis:\n",gbdis)
      fzdis = np.zeros([d-1])
      for i in range(k):
        if fit[i] != fit[j]:
          u[i] = (1 / (fit[i] - fit[j])) * (fit[j] / l)
          rk = np.linalg.norm(ppl[i] - ppl[j])
          fzdis += (u[i] * np.exp(-gamma * pow(rk,2))) * (ppl[i] - ppl[j])
      #print("fzdis:\n",fzdis)
      alpha = np.zeros(d-1)
      for m in range(d-1):
        alpha[m] = e_alp * (random.random() - 0.5)
      temp_ppl = ppl[j] + lbdis + gbdis + fzdis + alpha
      temp_ppl = sigmoid(temp_ppl, np.random.rand(d-1))
      if np.sum(temp_ppl) == 0:
        temp_ppl = np.random.choice(a=[1.0,0.0], size=(d-1), p=[p,1-p])
      temp_fit = np.array([inter_fitness(temp_ppl)])
      ppl[j] = temp_ppl
      fit[j] = temp_fit
      if temp_fit > fit[j]:
        lbest[j] = temp_ppl

    arr = np.hstack([ppl,fit,lbest])
    bestchoice = arr[np.argsort(arr[:,d-1])[::-1]]
    ppl = np.array(bestchoice[:,:d-1])
    fit = np.array(bestchoice[:,d-1:d])
    lbest = np.array(bestchoice[:,d:])
    rec = np.hstack([ppl,fit])
    bs = rec[0]
    print("\nBest of Iteration",e+1,":\n",bs)
    bestitr[e] = np.array(bs)
    with open(name, 'a',newline='\n') as out:
      writer = csv.writer(out,delimiter=',')
      rsl = np.concatenate([["Iter "+str(e+1)],bs])
      writer.writerow(rsl)
    out.close()


print("\nProcess Started")
d = 42
n = 10
k = 2
c1 = 100
c2 = 10000
itr = 30
lb = 0
ub = 1
alp_init = 0.2
l = 1000
gamma = 1
p = 0.7                                                               # probability of number of 1's in randomly generated 'ppl'
DS = pd.read_csv(r"D:\Users\juhis\PyCharm\FA\train_nslkdd_2class_preprocessed_normalized.csv")
data = np.array(DS)
training_data = np.delete(data, d-1, 1)                               # deletes d-1 th stream along axis 1 (columns) from 'data' and store into 'training_data'
target_data = np.delete(data, np.s_[:d-1], 1)                         # deletes first d-1 streams along axis 1 (columns) from 'data' and store into 'target_data'
name = "NSLKDD2classmodFAresult.csv"
ppl = np.random.choice(a=[1.0,0.0], size=(n,d-1), p=[p,1-p])          # generates 'ppl' having n*(d-1) sized array containg 1's and 0's
fit = init_fitness(ppl)                                               # calculates fitness of each row in 'ppl' and store into 'fit'
arr = np.hstack([ppl,fit])                                            # merges 'ppl' and its 'fit' and store into 'arr'
rec = arr[np.argsort(arr[:,-1])[::-1]]                                # 'np.argsort' gives indexes of sorted 'arr' , reverse into decreasing order , store into 'rec'
ppl = np.array(rec[:,:-1])                                            # stores each stream excluding stream's last element (fit) and rewrite into 'ppl'
fit = np.array(rec[:,-1:])                                            # stores each stream's last element (fit) and rewrite into 'fit'
lbest = np.zeros([n,d-1])
lbest = ppl
temp_best = np.zeros(d+1)
bestitr = np.zeros([itr, d])

toprow=[]
for l in range(0,d-1):
	toprow.append("Dim "+str(l+1))
with open(name, 'w',newline='\n') as op:
  writer = csv.writer(op,delimiter=',')
  header= np.concatenate([["Iter no."], toprow, ["Fitness"]])
  writer.writerow(header)
op.close()
modFA(k,rec,ppl,fit,lbest,bestitr)
final = bestitr[np.argsort(bestitr[:,-1])[::-1]]
print("\nGlobal Best:\n",final[0])
print("\nProcess Finished\n")