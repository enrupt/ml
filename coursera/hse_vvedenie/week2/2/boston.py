import numpy as np
from sklearn.datasets import load_boston
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score

data = load_boston()
lab_enc = preprocessing.LabelEncoder()
x = data.get('data')
y = data.get('target')
x = preprocessing.scale(x)

best_quality = -1000;
best_p = -1;

for p in np.linspace(1,10,200):
    neigh = KNeighborsRegressor(metric='minkowski', p=p, n_neighbors=5, weights='distance')
    kf = KFold(n_splits=5, shuffle=True,  random_state=42)
    quality = np.mean(cross_val_score(neigh, x, y, cv=kf, scoring='neg_mean_squared_error'))
    if (quality > best_quality) :
        best_quality = quality
        best_p = p

print(best_p)
print(best_quality)