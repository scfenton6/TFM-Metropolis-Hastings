"""
Comparison of our three symplectic integrators' performance when approximating
the path given by the system of ordinary differential equations dq/dt=p, dp/dt=-q,
whose analytical solution corresponds to the unit circle.
"""

import numpy as np
import matplotlib.pyplot as plt
from symplectic_integrators import *

def U(q):
    return q**2 / 2

def gradU(q):
    return q 
    
q, p, eps, m = 0, 1, 0.3, 1  # initial data

theta = np.linspace(-np.pi, np.pi, 100)
x =np.sin(theta) 
y = np.cos(theta)

eul_pos, eul_mom = sample_path(12, gradU, eps, euler, q, p, m)
eul_m_pos, eul_m_mom = sample_path(12, gradU, eps, euler_mod, q, p, m)
lf_pos, lf_mom = sample_path(12, gradU, eps, leapfrog, q, p, m)

plt.plot(x,y, label='analytic solution')
plt.plot(eul_pos, eul_mom, '--bo', color ='b', label='euler')
plt.plot(eul_m_pos, eul_m_mom, '--bo', color ='g', label='mod. euler')
plt.plot(lf_pos, lf_mom, '--bo', color ='r', label='leapfrog')
plt.axis('scaled')
plt.legend(fontsize=7)
plt.show()