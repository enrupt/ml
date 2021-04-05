import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sts


b = 2.62
pareto_rv = sts.pareto(b)
sample = sts.pareto.rvs(b, scale=0.8, size=1000)

x = np.linspace(1, 10, 100)
cdf = pareto_rv.pdf(x)
plt.plot(x, cdf, label='theoretical CDF')

# я хз что такое normed, вот тут
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.hist.html
# такого параметра нет, юзаем density
plt.hist(sample, bins=30, density=True)
plt.legend(loc='lower right')

mean, var = sts.pareto.stats(b)
print(mean, var)

samples = []
for n in (3, 5, 10, 50):
    sample_n = []
    for i in range(1, 1001):
        n_variates = sts.pareto.rvs(b, scale=0.8, size=n)
        sample_n.append(n_variates.mean())

    samples.append(sample_n)
    norm_rv = sts.norm(mean, var/n)
    pdf_norm = norm_rv.pdf(x)

    plt.figure()
    plt.plot(x, pdf_norm, label='Norm distribution of ' + str(n))

    plt.hist(sample_n, bins=20, density=True)