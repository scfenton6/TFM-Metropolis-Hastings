"""
Accept-Reject method for sampling from beta distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

n = 1000
a, b = 2.7, 6.3

def f(x):  # Beta distribution density
  numer = x**(a-1) * (1-x)**(b-1)
  denom = gamma(a) * gamma(b) /gamma(a+b)
  return 1/denom * numer
  
mode = (a-1)/(a+b-2)  # Beta density mode
m = f(mode)  # maximum value attained by Beta density

xx = np.linspace(0,1,500)

i = 0
total = 0
output = np.zeros(n)

while i < n:
  Y = np.random.uniform(low=0, high=1)
  U = np.random.uniform(low=0, high=1)
  if U < (1/m)*f(Y): 
      output[i]=Y
      i += 1
  total +=1
print("Empirical acceptance rate:", n/total)  
print("Theoretical acceptance rate=", 1/m)  
plt.plot(xx, f(xx))
plt.hist(output, bins=15, density=True)
plt.show()