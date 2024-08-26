import math
from scipy.optimize import differential_evolution, minimize


def f(x):
    return math.sin(x/5.0) * math.exp(x/10.0) + 5 * math.exp(-x/2.0)


def h(x):
    return int(f(x))


min1_1 = minimize(f, x0=2, method='BFGS')
print(min1_1)
min1_2 = minimize(f, x0=30, method='BFGS')
print(min1_2)

print('****************************************')

min2_1 = differential_evolution(f, [(1.0, 30.0)])
print(min2_1)
min2_2 = differential_evolution(h, [(1.0, 30.0)])
print(min2_2)

print('****************************************')

min3_2 = minimize(h, x0=30, method='BFGS')
print(min3_2)
min3_2 = differential_evolution(h, [(1.0, 30.0)])
print(min3_2)



