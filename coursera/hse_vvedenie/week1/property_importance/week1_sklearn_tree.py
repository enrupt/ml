import pandas as pd

from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('../titanic.csv', index_col='PassengerId')

properties = df.loc[:, ['Pclass', 'Fare', 'Age', 'Sex']]
properties = properties.dropna()
properties['Sex'] = properties['Sex'].map({"male": 0, "female": 1})

clf = DecisionTreeClassifier(random_state=241)
y = df["Survived"]
y = y[properties.index]

clf.fit(properties[['Pclass', 'Fare', 'Age', 'Sex']], y)
importances = clf.feature_importances_
print(importances)