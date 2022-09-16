from numpy.random import choice
import matplotlib.pyplot as plt
import math
  
def frecs_ehrenfest(n_iters = 1000, n_mol = 100):
    y_vals = [n_mol]  # initial númber of molecules in left urn
    frecs = [0]*(n_mol+1)
    frecs[-1] = 1
    molecules = [i for i in range(0,n_mol+1)]
    while len(y_vals) < n_iters:
        i = y_vals[-1] # current value of our Markov chain
        p0 = (n_mol - i)/n_mol  # transition probability from i to i+1
        p1 = i/n_mol  # transition probability from i to i-1
        y_step = choice([i+1,i-1], p=[p0,p1]) # new value for our Markov Chain
        frecs[y_step]+=1
        y_vals.append(y_step)
    rel_freq = [i/(sum(frecs)) for i in frecs]  # relative frequencies
    return molecules, rel_freq


def mu(M):
    mu = []
    for i in range(M):
        mu_i = math.comb(M, i)*(2**(-M))
        mu.append(mu_i)
    return mu

#time_data, pos_data = frecs_ehrenfest(n_iters = 1000)
time_data, pos_data = frecs_ehrenfest(n_iters = 10000)
plt.plot(time_data, pos_data, 'o', markersize=5)
plt.plot(time_data, mu(50), 'k-')

plt.title("Experimento de Ehrenfest")
#plt.savefig('ehrenfest_1000.pdf')
#plt.savefig('ehrenfest_10000.pdf')
plt.show()