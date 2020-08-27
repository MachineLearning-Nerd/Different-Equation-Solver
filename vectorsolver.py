import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Time in days
# t = 100

# Population of states
N = np.array([124, 237, 29, 85]) * 10*6

S0 = np.array([124, 237, 29, 85]) * 10*6
S0 = S0/N
E0 = np.array([11, 31, 5, 24])
E0 = E0/N
I0 = np.array([4, 38, 31, 15])
I0 = I0/N
R0 = np.array([0, 0, 0, 0])
R0 = R0/N
print(S0)
AvgR0 = np.array([1.88, 1.7, 1.62, 1.56])

d = 4
z = 5
# beta = x*y
gamma = 1/d
delta = 1/z
beta = AvgR0 * gamma
print(beta)
beta = np.diag(beta)
# def calculate_beta(R_value):
#     betaval = R_value * gamma
#     betaval = np.diag(betaval)
#     return betaval

# R0_vec = beta/gamma

# Equations
# dsdt = (-1) * np.matmul(beta, S0*I0)
# dedt = np.matmul(beta, S0*I0) - delta* E0
# didt = delta* E0 - gamma*I0
# drdt = gamma*I0

def derivative_fun(all_vectors, t0):
    # print("infunction")
    S = all_vectors[:4]
    E = all_vectors[4:8]
    I = all_vectors[8:12]
    R = all_vectors[12:]
    # import pdb; pdb.set_trace()
    # beta = calculate_beta(R)
    # print(beta)
    dsdt = (-1) * np.matmul(beta, S*I)
    dedt = np.matmul(beta, S*I) - delta* E
    didt = delta* E - gamma*I
    drdt = gamma*I

    # print("infunction")
    # print(dsdt, dedt, didt, drdt)

    derivative_vector = []
    derivative_vector.extend(dsdt)
    derivative_vector.extend(dedt)
    derivative_vector.extend(didt)
    derivative_vector.extend(drdt)

    return derivative_vector


t = np.arange(300)
full_vector = []
full_vector.extend(S0)
full_vector.extend(E0)
full_vector.extend(I0)
full_vector.extend(R0)
# Here we will get shape: (300, 16)
v = spi.odeint(derivative_fun, full_vector, t)

import os
os.makedirs('Images', exist_ok=True)
states = ['Bihar', 'UttarPradesh', 'Haryana', 'MadhyaPradesh']
for i in range(4):
    print(i, i+4, i+8, i+12)
    plt.plot(v[:, i])
    plt.plot(v[:, i+4])
    plt.plot(v[:, i+8])
    plt.plot(v[:, i+12])
    plt.title(f'Graph for {states[i]}')
    plt.legend(['susceptible', 'exposed', 'infectious', 'recovered'])
    plt.savefig(f'Images/{states[i]}.png')
    plt.show()
