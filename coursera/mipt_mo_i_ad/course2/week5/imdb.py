#!/usr/bin/env python
# coding: utf-8

# # Рецензии на imdb

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib', 'inline')


# Имеются 25000 рецензий пользователей imdb с бинарными метками, посчитанными по оценкам: 0 при оценке < 5 и 1 при оценке >=7.
# 
# Полные данные: https://www.kaggle.com/c/word2vec-nlp-tutorial/data
# 
# Загрузим выбоку:

# In[2]:


imdb = pd.read_csv('labeledTrainData.tsv', delimiter='\t')
imdb.shape


# In[3]:


imdb.head()


# Классы сбалансированы:

# In[4]:


imdb.sentiment.value_counts()


# Разобъём выборку на обучение и контроль:

# In[5]:


from sklearn.cross_validation import train_test_split
texts_train, texts_test, y_train, y_test = train_test_split(imdb.review.values, imdb.sentiment.values)


# Векторизуем тексты рецензий:

# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(sublinear_tf=True, use_idf=True)
X_train = vect.fit_transform(texts_train)
X_test = vect.transform(texts_test)


# ## Логистическая регрессия

# Настроим на векторизованных данных логистическую регрессию и посчитаем AUC:

# In[7]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
clf = LogisticRegression()
clf.fit(X_train, y_train)
print metrics.accuracy_score(y_test, clf.predict(X_test))
print metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


# Признаков получилось очень много:

# In[8]:


X_train.shape


# Попробуем отбирать признаки с помощью лассо:

# In[12]:


clf = LogisticRegression(C=0.15, penalty='l1')
clf.fit(X_train, y_train)
print np.sum(np.abs(clf.coef_) > 1e-4)
print metrics.accuracy_score(y_test, clf.predict(X_test))
print metrics.roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])


# Ещё один способ отбора признаков — рандомизированная логистическая регрессия:

# In[13]:


from sklearn.linear_model import RandomizedLogisticRegression
rlg = RandomizedLogisticRegression(C=0.13)
rlg.fit(X_train, y_train)


# Посмотрим, сколько признаков отбирается:

# In[14]:


np.sum(rlg.scores_ > 0)


# Настроим логистическую регрессию на отобранных признаках:

# In[15]:


X_train_lasso = X_train[:, rlg.scores_ > 0]
X_test_lasso = X_test[:, rlg.scores_ > 0]


# In[16]:


clf = LogisticRegression(C=1)
clf.fit(X_train_lasso, y_train)
print metrics.accuracy_score(y_test, clf.predict(X_test_lasso))
print metrics.roc_auc_score(y_test, clf.predict_proba(X_test_lasso)[:, 1])


# ## Метод главных компонент

# Сделаем 100 синтетических признаков с помощью метода главных компонент:

# In[17]:


from sklearn.decomposition import TruncatedSVD
tsvd = TruncatedSVD(n_components=100)
X_train_pca = tsvd.fit_transform(X_train)
X_test_pca = tsvd.transform(X_test)


# Обучим на них логистическую регрессию:

# In[18]:


clf = LogisticRegression()
clf.fit(X_train_pca, y_train)
print metrics.accuracy_score(y_test, clf.predict(X_test_pca))
print metrics.roc_auc_score(y_test, clf.predict_proba(X_test_pca)[:, 1])


# По 100 полученных таким способом признакам качество получается не намного хуже, чем по всем 66702!
# 
# Попробуем обучить на них обучить случайный лес:

# In[19]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train_pca, y_train)
print metrics.accuracy_score(y_test, clf.predict(X_test_pca))
print metrics.roc_auc_score(y_test, clf.predict_proba(X_test_pca)[:, 1])


# Признаки, которые даёт метод главных компонент, оптимальны для линейных методов, поэтому логистическая регрессия показывает результаты лучше, чем сложные нелинейные классификаторы.
