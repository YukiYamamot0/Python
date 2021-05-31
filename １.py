# -*- coding: utf-8 -*-
"""
Created on Fri May  7 17:09:59 2021

@author: yukin
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

mglearn.plots.plot_scaling()
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target,random_state=1)

print(X_train.shape)
print(X_test.shape)

