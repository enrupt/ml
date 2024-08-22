#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn.datasets as ds
from sklearn.model_selection import train_test_split
#from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial import distance

X, y = ds.load_digits(return_X_y=True)

(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, shuffle=False)


# In[28]:


import sys
def get_closest_idx(from_x):
    closest_dist = sys.maxsize
    idx = -1
    for i in range(0, len(X_train) - 1):
        dst = distance.euclidean(from_x, X_train[i])
        if dst < closest_dist:
            closest_dist = dst
            idx = i
    return idx

def predict():
    predicted = []
    for x_to_test in X_test:
        class_idx = get_closest_idx(x_to_test)
        predicted.append(y_train[class_idx])
    return predicted


# In[34]:



y_predicted = predict();

def calc_mismatches(y_predicted):
    mismatches = 0.0;
    for i in range(0, len(y_test)):
         if(y_predicted[i] != y_test[i]):
             mismatches = mismatches + 1;
    return mismatches


mismatches = calc_mismatches(y_predicted);
print(mismatches/len(y_test))

from sklearn.metrics import zero_one_loss
zero_one_loss(y_test, y_predicted)


# In[35]:


from sklearn.ensemble import RandomForestClassifier

regr = RandomForestClassifier(n_estimators=1000)
regr.fit(X_train, y_train)


print( zero_one_loss(y_test, regr.predict(X_test)) )


# In[ ]:




