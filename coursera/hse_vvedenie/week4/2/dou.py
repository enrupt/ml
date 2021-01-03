import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

df_cp = pd.read_csv('close_prices.csv', index_col=False)
X = df_cp.iloc[:, 1:]
pca = PCA(n_components=10)
reduced = pca.fit_transform(X)

# sum = 0
# for k in range(0,11):
#     sum = sum + pca.explained_variance_ratio_[k]
#     print(sum, k)

# first = reduced[:, 0]
# df_dji = pd.read_csv('djia_index.csv', index_col=False)
# print(np.corrcoef(x=first, y=df_dji.iloc[:, 1]))

max = 0
max_k = -1
for k in range(0, len(pca.components_[0])):
    avg = np.average(pca.components_[:, k])
    if(avg > max):
        max = avg
        max_k = k
print(max_k)

max = 0
max_k = -1
for k in range(0, len(pca.components_[0])):
    cmp = pca.components_[0][k]
    if(cmp > max):
        max = cmp
        max_k = k
print(max_k)
#

print( X.columns[ np.argmax( pca.components_[0] ) ] )