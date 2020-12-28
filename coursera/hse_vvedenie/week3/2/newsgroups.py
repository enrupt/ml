from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn import model_selection

import numpy as np

newsgroups = datasets.fetch_20newsgroups(
    subset='all',
    categories=['alt.atheism', 'sci.space']
)
y = newsgroups.target
X = newsgroups.data

vectorizer = TfidfVectorizer()
X_transformed = vectorizer.fit_transform(raw_documents=X)

grid = {'C': np.power(10.0, np.arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = svm.SVC(kernel='linear', random_state=241)
gs = model_selection.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X_transformed, y)
#print (gs.best_score_)
#print (gs.best_index_)
#print (gs.best_params_)

clf = svm.SVC(kernel='linear', C=1.0, random_state=241)
clf.fit(X_transformed, y)

coefs = abs(clf.coef_.todense().A1)
coefs = np.argsort(coefs)[-10:]
feature_mapping = vectorizer.get_feature_names()

words = []
for coef in coefs:
    words.append(feature_mapping[coef])

print(sorted(words))