# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:51:37 2021

@author: yukin
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

cancer = load_breast_cancer()
#in[9]
from sklearn.svm import SVC
X_train,X_test,y_train,y_test = train_test_split(cancer.data,cancer.target,
                                                 random_state=0)
#なんか値が違う、質問
svm = SVC(C=100)
svm.fit(X_train,y_train)
print("test set accuracy:{:2f}".format(svm.score(X_test,y_test)))

#in[10]
#0-1スケール変換で前処理
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#変換された訓練データで学習
svm.fit(X_train_scaled, y_train)

#変換されたテストセットでスコア計算
print("Scaled test set accurary: {:.2f}".format(svm.score(X_test_scaled,
                                                          y_test)))

#in[11]
#平均を０に分散を１に前処理
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

#変換されたデータで学習
svm.fit(X_train_scaled, y_train)

#変換されたテストセットでスコアを計算
print("SVM test accuracy:{:.2f}".format(svm.score(X_test_scaled, y_test)))