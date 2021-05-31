# -*- coding: utf-8 -*-
"""
Created on Wed May 26 19:27:16 2021

@author: yukin
"""


import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = load_breast_cancer()

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

#in[15]
from sklearn.decomposition import PCA
#データの最初の2つの主成分だけを維持する
pca = PCA(n_components=2)
#cancerデータセットにPCAモデルを適合
pca.fit(X_scaled)

#最初の２つの主成分に対してデータポイントを変換
X_pca = pca.transform(X_scaled)
print("Original shape:{}".format(str(X_scaled.shape)))
print("Reduced shape:{}".format(str(X_pca.shape)))

#in[16]
#第1主成分と第2主成分によるプロット。クラスごとに色分け
plt.figure(figsize=(8,8))
mglearn.discrete_scatter(X_pca[:,0],X_pca[:,1],cancer.target)
plt.legend(cancer.target_names,loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First princial component")
plt.ylabel("Second principal component")

#in[17]
print("PCA component shape: {}".format(pca.components_.shape))

#in[18]
print("PCA components:\n{}".format(pca.components_))

#in[19]
plt.matshow(pca.components_, cmap='viridis')
plt.yticks([0,1],["First component","Second component"])
plt.colorbar()
plt.xticks(range(len(cancer.feature_names)),
          cancer.feature_names,rotation=60,ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
