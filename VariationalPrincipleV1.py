import numpy as np
import random
#Constants
hbar = 1.0
kT = 1e-5
m_e = 0.511e6
E_gs = -13.6
e = 0.303
eps_0 = 1.0
dr = 3e-5
a0 = 2.7e-4
#--------------------------------------------
#Functions
def Diff(array):
    deriv = np.zeros(len(array))
    for i in range(0, len(array)):
        if i == 0:
            diff = (array[i + 1] - array[i]) / dr
            deriv[i] = diff
        elif 0 < i < len(array) - 1:
            diff = (array[i + 1] - array[i - 1]) / (2*dr)
            deriv[i] = diff
        elif i == len(array) - 1:
            diff = (array[i] - array[i - 1]) / dr
            deriv[i] = diff
    return deriv
def Norm(array):
    J_array = np.zeros(len(array))
    for i in range(len(array)):
        J_array[i] = ((dr*i)**2) * array[i]
    A = 1.0 / np.sqrt(4*np.pi * dr * np.dot(J_array, array))
    norm_array = A * array
    return norm_array
def Lap(array):
    Lap_array = np.zeros(len(array))
    darray = Diff(array)
    rsqr_darray = np.fromiter((darray[i] * (dr*i)**2 for i in range(len(darray))), float)
    drsqr_darray = Diff(rsqr_darray)
    for i in range(1, len(array)):
        Lap_array[i] = drsqr_darray[i] / ((dr*i)**2)
    return Lap_array
def Energy(array):
    A = array[0]
    B = (4*array[1] - array[2] - 3*array[0]) / (2 * dr)
    C = (array[2] + array[0] -2*array[1]) / (2*dr**2)
    R = dr
    #Hamiltonian:
    Lap_array = Lap(array)
    T = np.fromiter(((-hbar**2 / (2*m_e)) * Lap_array[i] for i in range(len(Lap_array))), float)
    V = np.zeros(len(array))
    for i in range(1, len(array)):
        V[i] = ((-e**2) / (4 * np.pi * eps_0 * (dr*i))) * array[i]
    expH = 0
    H = T + V
    for i in range(len(psi)):
        if i == 0:
            expH += (-e**2 / eps_0) * ( (A*R)**2/(2)  + (2 * A*B*R**3)/(3) + (A*C*R**4)/(2) + (B**2 * R**4)/(4) + (2*B*C*R**5)/(5) + (C**2 * R**6)/(6)) + ((-2*hbar**2*np.pi)/(m_e)) * ((A*B*R**2) + (2*A*C*R**3) + (2*B**2 * R**3)/(3) + (2*B*C*R**4)/(3) + (C*B*R**4)/(2) + (6*C**2 * R**5)/(5))
        elif i > 0:
            T_here = 4 * np.pi * dr * (dr * i)**2 * T[i] * array[i]
            V_here = 4 * np.pi * dr * (dr * i)**2 * V[i] * array[i]
            expH += T_here + V_here
    return expH
#--------------------------------------------
#Initialize Wavefunction
psi = np.zeros(100)
psi = np.fromiter((len(psi) - i for i in range(len(psi))), float)
#Normalize Wavefunction
npsi = Norm(psi)
#Initial Energy
expH = Energy(npsi)
#----------------------------------------------
#Metropolis Alg:
accept_1 = 0
accept_2 = 0
reject = 0
l = 0
while True:
    pre_perturbed_npsi = npsi * 1.0
    #Perturb Normalized Wavefunction
    k = random.randint(0, len(npsi) - 1)
    h_shift = random.uniform(-0.03 * npsi[k], 0.03 * npsi[k])
    w_shift = random.uniform(10.0, 100)
    for i in range(len(npsi)):
        npsi[i] = npsi[i] + h_shift * np.exp(-(i - k)**2 / w_shift**2)
    #Boundary conditions for r -> +inf
    npsi[len(npsi) - 1] = 0
    npsi[len(npsi) - 3] = 0.5 * npsi[len(npsi) - 4]
    npsi[len(npsi) - 2] = 0.5 * npsi[len(npsi) - 3]
    #Cusp Condition
    npsi[0] = npsi[1] / (1 - ((e**2 * dr * m_e) / (4 * np.pi)))
    #Normalize
    npsi_prime = Norm(npsi)
    #Energy
    expH_prime = Energy(npsi_prime)
    diff_H = expH_prime - expH
    #Metropolis
    if expH_prime < expH:
        #accept
        expH = expH_prime
        accept_1 += 1
        npsi = npsi_prime * 1.0
    elif expH < expH_prime:
        P = np.exp(-diff_H / kT)
        u = random.random()
        if P < u:
            #reject
            npsi = pre_perturbed_npsi * 1.0
            reject += 1
        elif u < P:
            #accept
            expH = expH_prime
            accept_2 += 1
            npsi = npsi_prime * 1.0
    l += 1
    accept_percent = ((accept_1 + accept_2) / (l)) * 100
    #print(l, "Current Energy:", expH, "eV")
    scaling = np.amax(npsi) / len(npsi)
    #Animate
    if l % 100 == 0:
        print("T -0.1 0.3")
        print("Energy:", expH, "eV")
        print("T -0.1 0.5")
        print("Iterations:", l)
        print("T -0.1 0.6")
        print("Total Type 1 Accepts:", accept_1, ";","Total Type 2 Accepts:", accept_2)
        print("T -0.1 0.4")
        print("Total Rejects:", reject)
        print("T -0.1 0.7")
        print("Acceptance Percentage:", accept_percent, "%")
        for q in range(len(npsi) - 1):
            print("l3", q, npsi[q] / scaling, 0, q + 1, npsi[q + 1] / scaling, 0)
            #print("c3", q, npsi[q] / scaling, 0, 0.25)
        print("F")
#print("accept1:", accept_1, "accept2:", accept_2)
