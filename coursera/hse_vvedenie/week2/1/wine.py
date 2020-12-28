import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing

df = pd.read_csv('wine.data', header=None, index_col=False)

best_quality = 0;
best_k = -1;

classes = df.loc[:, 0]
attrs = df.loc[:, 1:13]

for k in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5, shuffle=True,  random_state=42)
    quality = np.mean(cross_val_score(neigh, attrs, classes, cv=kf, scoring='accuracy'));
    #print("{} {}".format(quality, k))
    if (quality > best_quality) :
        best_quality = quality
        best_k = k

print("best {} {}".format(best_quality, best_k))

    # works as well:
    # for train_index, test_index in kf.split(df):
    #
    #     train_classes = df.iloc[train_index].loc[:, 0]
    #     train_attrs = df.iloc[train_index].loc[:, 1:13]
    #
    #     neigh.fit(train_attrs, train_classes)
    #
    #     test_classes = df.iloc[test_index].loc[:, 0]
    #     test_attrs = df.iloc[test_index].loc[:, 1:13]
    #     print("score: ", neigh.score(test_attrs, test_classes))

best_quality = 0;
best_k = -1;

attrs = preprocessing.scale(df.loc[:, 1:13])

for k in range(1,51):
    neigh = KNeighborsClassifier(n_neighbors=k)
    kf = KFold(n_splits=5, shuffle=True,  random_state=42)
    quality = np.mean(cross_val_score(neigh, attrs, classes, cv=kf, scoring='accuracy'));
    #print("{} {}".format(quality, k))
    if (quality > best_quality) :
        best_quality = quality
        best_k = k

print("best {} {}".format(best_quality, best_k))