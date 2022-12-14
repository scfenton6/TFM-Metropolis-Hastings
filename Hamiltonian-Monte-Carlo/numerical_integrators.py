import numpy as np

def euler(gradU, eps, q, p, m):
    """One step of the explicit Euler integrator"""
    p0 = p
    p = p - eps * gradU(q)  # momentum step
    q = q + eps * (p0 / m)  # position step
    return q, p

def euler_mod(gradU, eps, q, p, m):
    """One step of the symplectic Euler integrator"""
    p = p - eps * gradU(q)  # momentum step
    q = q + eps * (p/m)  # use new momentum to update position
    return q, p

def leapfrog(gradU, eps, q, p, m):
    """One step of the leapfrog symplectic integrator"""
    p = p - (eps/2) * gradU(q)  # 1st half step for momentum
    q = q + eps * (p/m)  # full step for position
    p = p - (eps/2) * gradU(q)  # 2nd half step for momentum
    return q, p

def sample_path(n_iters, gradU, eps, method_used, q, p, m):
    """Applies either of the former methods iteratively to
    construct an approximation of a hamiltonian path"""
    pos_path = [q]
    mom_path = [p]
    for i in range(n_iters):
        q, p = method_used(gradU, eps, q, p, m)
        pos_path.append(q)
        mom_path.append(p)
    return pos_path, mom_path