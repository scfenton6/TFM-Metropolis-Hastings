import matplotlib.pyplot as plt
from HMC import *


def my_U(theta):  # U(theta) = -log(f(theta)) 
    norm_theta = np.sqrt(np.sum(theta**2))
    return -20*(norm_theta-10)**2

def my_gradU(theta):  # gradient of U(theta)
    norm_theta = np.sqrt(np.sum(theta**2))
    return 40*(norm_theta-10)/norm_theta * theta

samples, acc_rate = HMC_path(
    n_iters=300, 
    U=my_U, 
    gradU=my_gradU, 
    eps=0.2, L=50, 
    q=np.array([3.0, 0.0]), 
    m=np.array([1.0, 1.0])
    )

print(acc_rate)
coord1, coord2 = zip(*samples[len(samples)//2:])  # pairs corresponding to the last half of the HMC algorithm iterations 
plt.plot(coord1, coord2)
plt.axis('scaled')
plt.show()