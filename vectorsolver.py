import numpy as np
import pandas as pd
import scipy.integrate as spi
import matplotlib.pyplot as plt

# Population of states
N = np.array([124, 237, 29, 85]) * 10**6

S0 = N
S0 = S0/N
E0 = np.array([11, 31, 5, 24])
E0 = E0/N
I0 = np.array([4, 38, 31, 15])
I0 = I0/N
R0 = np.array([0, 0, 0, 0])
R0 = R0/N
AvgR0 = np.array([1.88, 1.7, 1.62, 1.56])

gamma = 1/4
delta = 1/5
beta = AvgR0 * gamma
beta = np.diag(beta)

def derivative_fun(all_vectors, t0):
    S = all_vectors[:4]
    E = all_vectors[4:8]
    I = all_vectors[8:12]
    R = all_vectors[12:]

    dsdt = (-1) * np.matmul(beta, S*I)
    dedt = np.matmul(beta, S*I) - delta* E
    didt = delta* E - gamma*I
    drdt = gamma*I

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
print(v.shape)

all_data = pd.DataFrame()
import os
os.makedirs('Images', exist_ok=True)
states = ['Bihar', 'UttarPradesh', 'Haryana', 'MadhyaPradesh']

for i in range(4):
    # print(i, i+4, i+8, i+12)
    all_data[f"{states[i]}_s(t)"] = v[:, i]
    all_data[f"{states[i]}_e(t)"] = v[:, i+4]
    all_data[f"{states[i]}_i(t)"] = v[:, i+8]
    all_data[f"{states[i]}_r(t)"] = v[:, i+12]
    plt.plot(v[:, i])
    plt.plot(v[:, i+4])
    plt.plot(v[:, i+8])
    plt.plot(v[:, i+12])
    plt.title(f'Graph for {states[i]}')
    plt.xlabel('Time(Days)')
    plt.ylabel('Ratio of Individuals')
    plt.legend(['susceptible', 'exposed', 'infectious', 'recovered'])
    plt.savefig(f'Images/{states[i]}.png')
    plt.close()
print(all_data.shape)
print(all_data.head())
all_data.to_csv('all_state_data.csv')
