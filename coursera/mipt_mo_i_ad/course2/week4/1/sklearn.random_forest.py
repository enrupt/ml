#!/usr/bin/env python
# coding: utf-8

# # Sklearn

# ## sklearn.ensemble.RandomForestClassifier

# документация:  http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# In[2]:


get_ipython().run_line_magic('pylab', 'inline')


# In[21]:


from sklearn.model_selection import cross_validate
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

import numpy as np
import pandas as pd


# ### 1

# In[14]:


digits = datasets.load_digits()


# In[15]:


data = digits['data']
target = digits['target']


# In[20]:


dt_clf = DecisionTreeClassifier()
cross_val_score(dt_clf, data, digits.target, cv=10).mean()


# ### 2

# In[22]:


bag_clf = BaggingClassifier(base_estimator=dt_clf, n_estimators=100)
cross_val_score(bag_clf, digits.data, digits.target, cv=10).mean()


# ### 3

# In[29]:


d = (int) (math.sqrt(digits.data.shape[1]))


# In[31]:


bag_clf2 = BaggingClassifier(base_estimator=dt_clf, n_estimators=100, max_features=d)
cross_val_score(bag_clf2, digits.data, digits.target, cv=10).mean()


# ### 4

# In[43]:


dt_clf2 = DecisionTreeClassifier(splitter="random", max_features=d)
bag_clf3 = BaggingClassifier(base_estimator=dt_clf2, n_estimators=100)
cross_val_score(bag_clf3, digits.data, digits.target, cv=10).mean()


# In[ ]:





# In[ ]:




