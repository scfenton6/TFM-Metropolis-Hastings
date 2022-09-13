import numpy as np
from symplectic_integrators import *

def HMC_step(U, gradU, eps, L, q, m):
    
    """Stage 1: proposal generation"""
    U0 = U(q)  # current value for potential energy 
    q0 = np.copy(q)
    p = np.random.normal(loc=0.0, scale=1.0, size=len(q))
    K0 = sum(np.square(p)) / 2  # current value for kinetic energy 
    for i in range(L):
        q, p = leapfrog(gradU, eps, q, p, m)
    p = -p  # multiply momentum by -1 so proposal is reversible
    U = U(q)  # new value for potential energy 
    K = sum(np.square(p)) / 2  # new value for kinetic energy 
    
    """Stage 2: accept/reject proposal"""
    if np.random.uniform() < np.exp(U - U0 + K - K0):
        return q  # proposal is accepted
    
    else:
        return q0  # proposal is rejected

def HMC_path(n_iters, U, gradU, eps, L, q, m):
    """Constructs a Hamiltonian path approximation
    using the HMC_step function iteratively"""
    path = [q]
    accepted = 1
    for i in range(n_iters):
        old_q = np.copy(q)
        q = HMC_step(U, gradU, eps, L, q, m)
        if not np.array_equal(q, old_q):
            accepted+=1
        path.append(q)
    return path, accepted/n_iters  # return path and acceptance rate
