import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack
from sklearn.linear_model import Ridge

df_train = pd.read_csv('salary-train.csv', index_col=False)

df_train['FullDescription'] = df_train['FullDescription'].apply(str.lower)
tfid_vectorizer = TfidfVectorizer(min_df=5)
X_train_desc = tfid_vectorizer.fit_transform(df_train['FullDescription'])

df_train['LocationNormalized'].fillna('nan', inplace=True)
df_train['ContractTime'].fillna('nan', inplace=True)
dict_vectorizer = DictVectorizer()
dict_processed_train = dict_vectorizer.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_desc, dict_processed_train])
y_train = df_train['SalaryNormalized']


df_test = pd.read_csv('salary-test-mini.csv', index_col=False)
df_test['FullDescription'] = df_test['FullDescription'].apply(str.lower)
df_test['FullDescription'] = df_test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex = True)

X_test_desc = tfid_vectorizer.transform(df_test['FullDescription'])
df_test['LocationNormalized'].fillna('nan', inplace=True)
df_test['ContractTime'].fillna('nan', inplace=True)
dict_processed_test = dict_vectorizer.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_test = hstack([X_test_desc, dict_processed_test])

model = Ridge(alpha=1.0, random_state=241)
model.fit(X_train, y_train)
print(model.predict(X_test))
