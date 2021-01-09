import inline as inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('gbm-data.csv', index_col=False)
df_values = df.values

X = df_values[:, 1:]
y = df_values[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)

def sigmoid(_y):
    return 1.0 / (1.0 + np.exp(0.0 - _y))

#rate_arr = [1, 0.5, 0.3, 0.2, 0.1]
rate_arr = [0.2]
for rate in rate_arr:
    gbcl = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=rate)
    gbcl.fit(X_train, y_train)
    prediction_train = gbcl.staged_decision_function(X_train)
    train_score = np.empty(len(gbcl.estimators_))
    for i, y_pred_train in enumerate(gbcl.staged_decision_function(X_train)):
        #print("y_pred_train", y_pred_train, y_train)
        train_score[i] = log_loss(y_train, sigmoid(y_pred_train))

    # gbcl = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=rate)
    # gbcl.fit(X_test, y_test)
    prediction_train = gbcl.staged_decision_function(X_test)
    test_score = np.empty(len(gbcl.estimators_))
    min_loss = 1000000000
    min_i = -1
    for i, y_pred_test in enumerate(gbcl.staged_decision_function(X_test)):
        #print("y_pred_test", y_pred_test, y_test)
        test_score[i] = log_loss(y_test, sigmoid(y_pred_test))
        if(test_score[i] < min_loss):
            min_loss = test_score[i]
            min_i = i

    # plt.figure()
    # plt.plot(train_score, 'r', linewidth=2)
    # plt.plot(test_score, 'g', linewidth=2)
    # plt.legend(['test', 'train'])
    # plt.show()

    print(min_loss, min_i)

regr = RandomForestClassifier(n_estimators=36, random_state=241)
regr.fit(X_train, y_train)
print( log_loss(y_test, regr.predict_proba(X_test)) )