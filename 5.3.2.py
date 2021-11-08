# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 14:00:55 2021

@author: yukin
"""

import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import mglearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.svm import SVC

digits = load_digits()
y = digits.target == 9

X_train,X_test,y_train,y_test = train_test_split(
    digits.data,y,random_state=0)

from sklearn.dummy import DummyClassifier
dummy_majority = DummyClassifier(strategy = 'most_frequent').fit(X_train,y_train)
pred_most_frequent = dummy_majority.predict(X_test)
print("Unique predicted labels: {}".format(np.unique(pred_most_frequent)))
print("Test score: {:.2f}".format(dummy_majority.score(X_test,y_test)))

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=2).fit(X_train,y_train)
pred_tree = tree.predict(X_test)
print("Test score {:.2f}".format(tree.score(X_test,y_test)))

dummy = DummyClassifier().fit(X_train,y_train)
pred_dummy = dummy.predict(X_test)
print("dummy score:{:.2f}".format(dummy.score(X_test,y_test)))

logreg = LogisticRegression(C=0.1).fit(X_train, y_train)
pred_logreg = logreg.predict(X_test)
print("logreg score: {:.2f}".format(logreg.score(X_test,y_test)))

from sklearn.metrics import confusion_matrix

confusion = confusion_matrix(y_test, pred_logreg)
print("Confusion matrix:\n{}".format(confusion))

#mglearn.plots.plot_confusion_matrix_illustration()

#mglearn.plots.plot_binary_confusion_matrix()

print("Most frequenr class:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nDummy model:")
print(confusion_matrix(y_test, pred_most_frequent))
print("\nDecision trre:")
print(confusion_matrix(y_test, pred_tree))
print("\nLogistic Regression")
print(confusion_matrix(y_test,pred_logreg))

from sklearn.metrics import f1_score
print("f1 score most frequent: {:2f}".format(
    f1_score(y_test,pred_most_frequent)))
print("f1 score dummy: {:.2f}".format(f1_score(y_test, pred_dummy)))
print("f1 score tree: {:.2f}".format(f1_score(y_test, pred_tree)))
print("f1_score logistic regression: {:.2f}".format(
    f1_score(y_test, pred_logreg)))


from sklearn.metrics import classification_report
print(classification_report(y_test, pred_most_frequent,
                            target_names =["not nine","nine"]))

print(classification_report(y_test, pred_dummy,
                            target_names=["not nine","nine"]))

print(classification_report(y_test, pred_logreg,
                            target_names=["not nine","nine"]))

from mglearn.datasets import make_blobs
X, y = make_blobs(n_samples=(400, 50),cluster_std=[7.0,2],random_state=22)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
svc = SVC(gamma=.05).fit(X_train,y_train)

mglearn.plots.plot_decision_threshold()

print(classification_report(y_test,svc.predict(X_test)))

y_pred_lower_threshold = svc.decision_function(X_test) > -.8

print(classification_report(y_test, y_pred_lower_threshold))

from sklearn.metrics import precision_recall_curve
precision, recall, thresholds = precision_recall_curve(
    y_test, svc.decision_function(X_test))

#カーブがなめらかになるようにデータポイントを増やす
X,y=make_blobs(n_samples=(4000,500),cluster_std=[7.0, 2],random_state=22)
X_train, X_test, y_train, y_test= train_test_split(X,y,random_state=0)
svc = SVC(gamma=.05).fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(
        y_test, svc.decision_function(X_test))
close_zero = np.argmin(np.abs(thresholds))
plt.plot(precision[close_zero], recall[close_zero],'o',markersize=10,
         label="threshold zero", fillstyle="none",c="k",mew=2)

plt.plot(precision,recall,label="precision recall curve")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")
#10秒くらい時間がかかった

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=0, max_features=2)
rf.fit(X_train,y_train)

#RandomForestClassifierにはpredict_probaはあるがdecision_functionにはない
precision_rf, recall_rf, thresholds_rf=precision_recall_curve(
    y_test, rf.predict_proba(X_test)[:,1])

plt.plot(precision, recall, label="svc")

plt.plot(precision[close_zero],recall[close_zero],'o',markersize=10,
         label="threshold zero svc", fillstyle ="none",c='k', mew=2)

plt.plot(precision_rf, recall_rf, label="rf")

close_defeault_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_defeault_rf],recall_rf[close_defeault_rf],'^',c='k',
         markersize=10, label="threshold 0.5 rf", fillstyle="none",mew=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")


print("f1_score of random forest: {:.3f}".format(
    f1_score(y_test, rf.predict(X_test))))
print("f1_score of svc: {:.3}".format(f1_score(y_test, svc.predict(X_test))))

from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])
ap_svc = average_precision_score(y_test, svc.decision_function(X_test))
print("Average precision of random forest: {:.3f}".format(ap_rf))
print("Average precision of svc: {:.3f}".format(ap_svc))
