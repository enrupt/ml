import numpy as np
import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv('abalone.csv', index_col=False)
df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

y = df['Rings']
X = df.iloc[:, 0:len(df.columns) - 1]
print(X)
scorer = make_scorer(r2_score)

for k in range(1,51):
    regr = RandomForestRegressor(n_estimators=k, random_state=1)
    regr.fit(X, y)
    kf = KFold(n_splits=5, shuffle=True,  random_state=1)
    print(k, cross_val_score(regr, X=X, y=y, cv=kf, scoring='r2').mean() )
