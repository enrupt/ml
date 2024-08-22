import numpy as np
from sklearn.svm._libsvm import cross_validation


def f(k):
    return pow(3, k) * np.math.exp(-3)/ np.math.factorial(k);

p = 1;
for k in range(0, 5):
    p -= f(k);

cross_validation
print(p);