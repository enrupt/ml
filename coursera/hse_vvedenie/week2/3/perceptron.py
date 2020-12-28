from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


df_train = pd.read_csv('perceptron-train.csv', header=None, index_col=False)
df_test = pd.read_csv('perceptron-test.csv', header=None, index_col=False)

X_train = df_train.loc[:, 1:]
y_train = df_train.loc[:, 0]

X_test = df_test.loc[:, 1:]
y_test = df_test.loc[:, 0]

print(X_train)
print(y_train)

clf = Perceptron()
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

before_scale = accuracy_score(y_test, predictions)
print(before_scale)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
clf = Perceptron()
clf.fit(X_train_scaled, y_train)
X_test_scaled = scaler.transform(X_test)
predictions = clf.predict(X_test_scaled)

after_scale = accuracy_score(y_test, predictions)
print(after_scale - before_scale)


