import pandas as pd
import sklearn.metrics
import numpy as np

df = pd.read_csv('data-logistic.csv', header=None, index_col=False)
y = df.loc[:, 0]
X = df.loc[:, 1:]
n = len(df)

def w1_next(w1, w2, c, k):
    sum_ = 0
    for i in range(n):
        exp = np.exp(-y[i] * (w1 * X.loc[i, 1] + w2 * X.loc[i, 2]))
        add = y[i] * X.loc[i, 1] * (1.0 - 1.0 / (1.0 + exp))
        sum_ += add
    return w1 + (k / n) * sum_ - k * c * w1


def w2_next(w1, w2, c, k):
    sum_ = 0
    for i in range(n):
        exp = np.exp(-y[i] * (w1 * X.loc[i, 1] + w2 * X.loc[i, 2]))
        add = y[i] * X.loc[i, 2] * (1.0 - 1.0 / (1.0 + exp))
        sum_ += add
    return w2 + (k / n) * sum_ - k * c * w2


epsilon = 10 ** -5

w1 = 10
w2 = 10
dist = 10
predictions = []
i = 0

while dist > epsilon:
    w1_n = w1_next(w1, w2, 10, 0.1)
    w2_n = w2_next(w1, w2, 10, 0.1)
    dist = np.sqrt(np.square(w1_n - w1) + np.square(w2_n - w2))
    w1 = w1_n
    w2 = w2_n

for idx in range(n):
    predictions.append(1.0 / (1.0 + np.exp(-w1*X.loc[idx, 1] - w2*X.loc[idx, 2])))

print (sklearn.metrics.roc_auc_score(y, predictions))

dist = 10
w1 = 0
w2 = 0
predictions = []
idx = 0

while dist > epsilon:
    w1_n = w1_next(w1, w2, 0, 0.2)
    w2_n = w2_next(w1, w2, 0, 0.2)
    dist = np.sqrt(np.power(w1_n - w1, 2) + np.power(w2_n - w2, 2))
    w1 = w1_n
    w2 = w2_n

for idx in range(n):
    predictions.append(1.0 / (1.0 + np.exp(-w1*X.loc[idx, 1] - w2*X.loc[idx, 2])))

print (sklearn.metrics.roc_auc_score(y, predictions))