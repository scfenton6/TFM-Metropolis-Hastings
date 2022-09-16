"""
Code for sampling from the Gamma distribution using
the independent Metropolis Hastings algorithm. For
more info about the underlying theory, see Monte Carlo 
Statistical Methods, by C.P. Robert and George Casella,
Chapter 7.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma

alpha, beta = 4.3, 1
a = math.floor(alpha)
b=a/alpha

def densGamma(x, alpha, beta):  # Gamma distribution density
  val = (beta**alpha / gamma(alpha))*(x**(alpha-1))*math.exp(-beta*x)
  return val

EV = (alpha/beta)**2 + alpha/(beta**2)  # analytical expresion for the expectation

def f(x):
    return densGamma(x, alpha, beta)

def g(x):
    return densGamma(x, a, b)

X = np.random.gamma(shape = alpha)  # initial state of our Markov chain
n = 10000
MC = []
means = []
sums = X**2 
accepted = 0 
for i in range(n):
  Y = np.random.gamma(shape = a, scale=1/b) 
  rho = min((f(Y)*g(X))/(f(X)*g(Y)),1) if f(Y)*g(X)>0 else 0 
  X = np.random.choice(a=[Y,X], p=[rho,1-rho])
  MC.append(X)
  sums += X**2
  means.append(sums/(i+1))
  accepted = accepted + 1 if X==Y else accepted

"""
Here we plot the approximations of the expectation given by Independent 
Metropolis-Hastings (in blue) against the analytical expectation (in red).
We can see the approximations rapidly converging to the expectation.
"""
plt.plot(means)
plt.ylim([18, 27])
plt.axhline(y=EV, color='r', linestyle='-', linewidth=0.7)
print("Acceptance rate:", (accepted/n)*100)
plt.xlim([0, 10000])
plt.show()