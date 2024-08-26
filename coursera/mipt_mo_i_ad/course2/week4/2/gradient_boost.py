from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

X, y = datasets.load_boston(return_X_y=True)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.25, shuffle=False)

base_trees = []
coeffs = []
for k in range(1,51):
    regressor = DecisionTreeRegressor(random_state=42, max_depth=5)
    regressor.fit(X_train, y_train)
    base_trees.append(regressor)
    coeffs.append(0.9)

def gbm_predict(X):
    return [sum([coeff * algo.predict([x])[0] for algo, coeff in zip(base_trees, coeffs)]) for x in X]

