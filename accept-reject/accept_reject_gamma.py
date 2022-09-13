import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import gamma

def sample_gamma(a,b):
    samples_exp=[]
    for i in range(a):
        X = np.random.exponential(scale=1.0)
        samples_exp.append(X)
    return b*sum(samples_exp)
    
alpha_0, beta_0 = 4.3, 1
a = math.floor(alpha_0)
b = a/alpha_0
M = b**(-a) * ((alpha_0-a)/((1-b)*math.exp(1)))**(alpha_0-a)

def f(x, alpha, beta):  # Gamma distribution density
  val = (beta**alpha / gamma(alpha))*(x**(alpha-1))*math.exp(-beta*x)
  return val

xx = np.linspace(0,11,10000)
ff = [f(i, alpha_0, beta_0) for i in xx]

n = 1000
i = 0
total = 0
output = np.zeros(n)

while i< n:
  Y = sample_gamma(a,b)
  U = np.random.uniform(low=0, high=1)
  if U <= f(Y, alpha_0, beta_0)/(M*f(Y, a, b)): 
      output[i]=Y
      i += 1
  total += 1

print("Empirical acceptance rate:", n/total)
print("Theoretical acceptance rate=", 1/M)
plt.plot(xx, ff)
plt.hist(output, bins=30, density=True)
plt.show()