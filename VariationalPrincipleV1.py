import numpy as np
import random

#Constants
hbar = 1.0
kT = 1.0
m_e = 0.5e6
E_gs = -13.6
e = 0.303
eps_0 = 1.0

h = 1.0
a = 1.0
b = 1001.0
x = np.linspace(a, b, int((b - a) / h))

#--------------------------------------------
#Functions

def Diff(array):
    deriv = np.zeros(len(array))
    for i in range(0, len(array)):
        if i == 0:
            diff = (array[i + 1] - array[i]) / h
            deriv[i] = diff
        elif 0 < i < len(array) - 1:
            diff = (array[i + 1] - array[i - 1]) / (2*h)
            deriv[i] = diff
        elif i == len(array) - 1:
            diff = (array[i] - array[i - 1]) / h
            deriv[i] = diff
    return deriv

def Norm(array):
    Jac_times_array = [4 * np.pi * (h*i)**2 * array[i] for i in range(len(array))]
    A = 1.0 / np.sqrt(np.dot(Jac_times_array, array))
    array = [A * array[i] for i in range(len(array))]
    return array

def Lap(array):
    Lap_array = np.zeros(len(array))
    darray = Diff(array)
    rsqr_darray = [darray[i] * (h*i)**2 for i in range(len(darray))]
    drsqr_darray = Diff(rsqr_darray)
    for i in range(1, len(array)):
        Lap_array[i] = drsqr_darray[i] / ((h*i)**2)
    return Lap_array

def ExpHamiltonian(psi, Ham_psi):
    Jac_psi = [4* np.pi * (h*i)**2 * psi[i] for i in range(len(psi))]
    ExpecHam = np.dot(Jac_psi, Ham_psi)
    return ExpecHam
    
#--------------------------------------------

#Initialize Wavefunction
psi = np.zeros(len(x))
psi = [2.0 for i in range(len(psi))]

#Normalize Wavefunction
npsi = Norm(psi)
    
#Hamiltonian Calculation
Lap_psi = Lap(npsi)
T = [(-hbar**2 / (2*m_e)) * Lap_psi[i] for i in range(len(Lap(npsi)))]
V = np.zeros(len(npsi))
for i in range(1, len(npsi)):
    V[i] = ((-e**2) / (4 * np.pi * eps_0 * (h*i))) * npsi[i]

#Expectation of H
H = np.add(T, V)
expH = ExpHamiltonian(psi, H)

#Metropolis Alg:
accept = 0
reject = 0
for l in range(1, int(1e2)):
    print(expH)
    print("Iteration:", l)
    k = random.randint(0, len(npsi) - 1)
    j = npsi[k]
    print("Before:", "k:", k, "psi[k]:", j)
    r = random.uniform(-1e-5, 1e-5)
    print("r", r, npsi[k] + r)
    npsi[k] = npsi[k] + r
    print("After:", "psi[k]:", npsi[k])
    
    npsi_prime = Norm(npsi)
    
    Lap_psi = Lap(npsi_prime)
    T = [(-hbar**2 / (2*m_e)) * Lap_psi[i] for i in range(len(Lap(npsi_prime)))]
    V = np.zeros(len(npsi_prime))
    for i in range(1, len(npsi_prime)):
        V[i] = ((-e**2) / (4 * np.pi * eps_0 * (h*i))) * npsi_prime[i]
        
    H = np.add(T, V)
    expH_prime = ExpHamiltonian(npsi_prime, H)
    diff_H = expH_prime - expH
    print("Energy:", expH_prime)
    
    if expH_prime < expH:
        accept
        print("~~~~accept 1~~~~~")
        print("expH:", expH, "expH_prime", expH_prime)
        expH = expH_prime
        accept += 1
        npsi = npsi_prime
    elif expH < expH_prime:
        P = np.exp(-diff_H / kT)
        u = random.random()
        if P < u:
            reject
            npsi[k] = j
            print("u:", u, "P:", P)
            print("~~~~~~reject~~~~~~~")
            reject += 1
        elif u < P:
            accept
            print("u:", u, "P:", P)
            print("~~~~~~accept 2~~~~~~")
            expH = expH_prime
            accept += 1
            npsi = npsi_prime
     
    print("-----------------")

print("accepts:", accept, "rejects:", reject)





