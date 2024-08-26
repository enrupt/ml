#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sklearn.datasets as ds

X_digits, y_digits = ds.load_digits(return_X_y=True)
X_breast_cancer, y_breast_cancer = ds.load_breast_cancer(return_X_y=True)


# In[16]:


from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import cross_val_score

clf_bern = BernoulliNB()
cv_bern_digits = cross_val_score(clf_bern, X_digits, y_digits)
cv_bern_breast_cancer = cross_val_score(clf_bern, X_breast_cancer, y_breast_cancer)
print(cv_bern_digits.mean())
print(cv_bern_breast_cancer.mean())


# In[17]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

clf_mn = MultinomialNB()
cv_mn_digits = cross_val_score(clf_mn, X_digits, y_digits)
cv_mn_breast_cancer = cross_val_score(clf_mn, X_breast_cancer, y_breast_cancer)
print(cv_mn_digits.mean())
print(cv_mn_breast_cancer.mean())


# In[18]:


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score

clf_gauss = GaussianNB()
cv_gauss_digits = cross_val_score(clf_gauss, X_digits, y_digits)
cv_gauss_breast_cancer = cross_val_score(clf_gauss, X_breast_cancer, y_breast_cancer)
print(cv_gauss_digits.mean())
print(cv_gauss_breast_cancer.mean())


# In[ ]:




