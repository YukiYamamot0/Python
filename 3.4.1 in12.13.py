# -*- coding: utf-8 -*-
"""
Created on Wed May 26 13:37:55 2021

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

#in[12]
mglearn.plots.plot_pca_illustration()
#in[13]
fig, axes=plt.subplots(15,2,figsize=(10,20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]

ax = axes.ravel()
    
     
for i in range(30):
    _, bins = np.histogram(cancer.data[:,i],bins=50)
    ax[i].hist(malignant[:,i], bins=bins,color=mglearn.cm3(0),alpha=.5)
    ax[i].hist(benign[:,i], bins=bins,color=mglearn.cm3(2),alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
    ax[0].set_xlabel("Feature manitude")
    ax[0].set_ylabel("Frequency")
    ax[0].legend(["malignant","begin"], loc="best")
    fig.tight_layout


