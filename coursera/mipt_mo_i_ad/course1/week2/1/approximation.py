from matplotlib import pylab as plt
import numpy as np
import scipy.linalg as la


def f(x):
    return np.sin(x / 5) * np.exp(x / 10) + 5 * np.exp(-x / 2)


def a1(x):
    return 3.43914511 - 0.18692825 * x


def a2(x):
    return 3.32512949 - 0.06531159 * x - 0.00760104 * x * x


def a3(x):
    return 4.36264154 - 1.29552587 * x + 0.19333685 * x * x - 0.00823565 * x * x * x


a_1 = np.array([[1, 1], [1, 15]])
b_1 = np.array([f(1), f(15)])
w_1 = la.solve(a_1, b_1)

print(w_1)

a_2 = np.array([[1, 1, 1], [1, 8, 64], [1, 15, 225]])
b_2 = np.array([f(1), f(8), f(15)])
w_2 = la.solve(a_2, b_2)

print(w_2)

a_3 = np.array([[1, 1, 1, 1], [1, 4, 16, 64], [1, 10, 100, 1000], [1, 15, 225, 3375]])
b_3 = np.array([f(1), f(4), f(10), f(15)])
w_3 = la.solve(a_3, b_3)

print(w_3)

x = np.arange(0, 15, 0.1)
plt.plot(x, f(x))
plt.plot(x, a1(x))
plt.plot(x, a2(x))
plt.plot(x, a3(x))

plt.show()
