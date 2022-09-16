
"""
Code for sampling from the Normal distribution via the random walk 
Metropolis Hastings algorithm, with proposals following a uniform 
distribution with parameter 1. For more info about the underlying 
theory, see Monte Carlo Statistical Methods, by C.P. Robert and 
George Casella, Chapter 7.
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

def MHRW(n, delta):
    MC = np.zeros(n)
    X = 0 
    for i in range(n):
      Y = X + np.random.uniform(low = -delta, high = delta) 
      rho = min(np.exp((X**2-Y**2)/2),1) 
      X = np.random.choice(a=[Y,X], p=[rho,1-rho])
      MC[i] = X
    return MC   

sample1 = MHRW(2500,1)
xx = np.linspace(-4, 4, 100)
plt.plot(xx, stats.norm.pdf(xx, 0, 1))
plt.hist(sample1, bins=30, density=True)

"""plot of the last 500 iterations of the random walk we constructed"""
#plt.plot(list(range(2000,2500)),sample1[2000:2500])  

plt.show()