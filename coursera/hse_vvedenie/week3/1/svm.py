from sklearn import svm
import pandas as pd

df_train = pd.read_csv('svm-data.csv', header=None, index_col=False)
y = df_train.loc[:, 0]
X = df_train.loc[:, 1:]

clf = svm.SVC(kernel='linear', C=100000, random_state=241)
clf.fit(X, y)
print(clf.support_)