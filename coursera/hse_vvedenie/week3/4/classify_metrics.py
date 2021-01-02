import pandas as pd
import sklearn.metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import numpy as np

df = pd.read_csv('classification.csv', index_col=False)

y = df.loc[:, 'true']
X = df.loc[:, 'pred']

tp = df[(df['true'] == 1) & (df['pred'] == 1)].iloc[:,0].count()
tn = df[(df['true'] == 0) & (df['pred'] == 0)].iloc[:,0].count()
fn = df[(df['true'] == 1) & (df['pred'] == 0)].iloc[:,0].count()
fp = df[(df['true'] == 0) & (df['pred'] == 1)].iloc[:,0].count()

print("", tp, fp, fn, tn)

acc = sklearn.metrics.accuracy_score(y, X)
pr = sklearn.metrics.precision_score(y, X)
rec = sklearn.metrics.recall_score(y, X)
f1 = sklearn.metrics.f1_score(y, X)

print("", acc, pr, rec, f1)

df2 = pd.read_csv('scores.csv', index_col=False)
y = df2.loc[:, 'true']
X_score_logreg = df2.loc[:, 'score_logreg']
X_score_svm = df2.loc[:, 'score_svm']
X_score_knn = df2.loc[:, 'score_knn']
X_score_tree = df2.loc[:, 'score_tree']

ra_LR = roc_auc_score(y, X_score_logreg)
ra_SVM = roc_auc_score(y, X_score_svm)
ra_KNN = roc_auc_score(y, X_score_knn)
ra_TR = roc_auc_score(y, X_score_tree)

print("", ra_LR, ra_SVM, ra_KNN, ra_TR)


def max_precision(_data):

    ext_thr = np.append(_data[2], 0)
    d = {'precision': _data[0], 'recall': _data[1], 'thresholds': ext_thr}
    pr_df = pd.DataFrame(data=d)

    pr_above_07 = pr_df[pr_df['recall'] >= 0.7]
    #print(pr_above_07)
    return pr_above_07.iloc[:, 0].max()

print(max_precision(precision_recall_curve(y, X_score_logreg)))
print(max_precision(precision_recall_curve(y, X_score_svm)))
print(max_precision(precision_recall_curve(y, X_score_knn)))
print(max_precision(precision_recall_curve(y, X_score_tree)))