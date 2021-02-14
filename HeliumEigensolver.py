import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import matplotlib.pyplot as plt

#Constants
N = int(2e3)
Z = 2
dr = 1.0e-6
hbar = 1.0
e = 0.303
m_e = 0.511e6
eps_0 = 1
n = 0

#---------------------------------
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

def Kinetic(array):
    array = diags([-2, 1, 1], [0, -1, 1], shape = [N,N]).toarray()
    array = (-hbar**2)/(2 * m_e * dr**2) * array
    return array
    
def Potential(array):
    r_0 = 1e-6
    for i in range(1, N):
        for j in range(1, N):
            if i == j:
                array[i, j] = -1.0 / (i * dr)
    array[0,0] = ((-1.0/r_0) + (1.0/r_0**2) * (0 - r_0) - (1/2)*(1/r_0**3)*(0 - r_0)**2)
    array = (Z*e**2 / (4*np.pi*eps_0)) * array
    return array
    
def V_Coul(array):
    r_0 = 1e-8
    for i in range(N):
        for j in range(N):
            if i != j:
                array[i, j] = 1.0 / np.abs((i - j) * dr)
            elif i == j:
                array[i, j] = ((-1.0/r_0) + (1.0/r_0**2) * (i - r_0) - (1/2)*(1/r_0**3)*(i - r_0)**2)
    array = e**2 / (4*np.pi*eps_0) * array
    return array
    
def Energy(array):
    
    #Taylor Expand Wavefunction near 0
    A = array[0]
    B = (4*array[1] - array[2] - 3*array[0]) / (2 * dr)
    C = (array[2] + array[0] -2*array[1]) / (2*dr**2)
    R = dr
    
    #Hamiltonian:
    Lap_array = Lap(array)
    T = np.fromiter(((-hbar**2 / (2*m_e)) * Lap_array[i] for i in range(len(Lap_array))), float)
    V = np.zeros(len(array))
    for i in range(1, len(array)):
        V[i] = ((-Z*e**2) / (4 * np.pi * eps_0 * (dr*i))) * array[i]
    expH = 0
    H = T + V
    for i in range(len(array)):
        if i == 0:
            expH += (-Z*e**2 / eps_0) * ( (A*R)**2/(2)  + (2 * A*B*R**3)/(3) + (A*C*R**4)/(2) + (B**2 * R**4)/(4) + (2*B*C*R**5)/(5) + (C**2 * R**6)/(6)) + ((-2*hbar**2*np.pi)/(m_e)) * ((A*B*R**2) + (2*A*C*R**3) + (2*B**2 * R**3)/(3) + (2*B*C*R**4)/(3) + (C*B*R**4)/(2) + (6*C**2 * R**5)/(5))
        elif i > 0:
            T_here = 4 * np.pi * dr * (dr * i)**2 * T[i] * array[i]
            V_here = 4 * np.pi * dr * (dr * i)**2 * V[i] * array[i]
            expH += T_here + V_here
    return expH
    
def U_eff(array, particle):
    #Hartree pseudo-potential
    V_Hart = np.zeros(len(array[particle]))
    r_0 = 1.0e-6
    for r in range(0, len(array[particle])):
        if r == 0:
            e**2*((array[particle][r])**2 + (array[np.abs(particle - 1)][r])**2) * (r*dr)**2 * dr * ((-1.0/r_0) + (1.0/r_0**2) * (0 - r_0) - (1/2)*(1/r_0**3)*(0 - r_0)**2)
        if r > 0:
            V_Hart[r] = V_Hart[r - 1]
            V_Hart[r] = V_Hart[r] + e**2*((array[particle][r])**2 + (array[np.abs(particle - 1)][r])**2) * (r*dr)**2 * dr / max(len(array[particle])*dr, len(array[np.abs(particle - 1)])*dr)
    V_Hart = V_Hart * (4*np.pi / eps_0)

    #Expectation of Coulomb (Hartree Approx.) Potential
    expU_eff = 0
    for i in range(len(array[particle])):
        expU_eff += 4 * np.pi * dr * (dr * i)**2 * V_Hart[i] * array[particle][i]**2 / len(array)
    return expU_eff
    
    
    
#----------------------------------
init_array = np.zeros([N,N], dtype = float)


#Hamiltonian
T_1 = T_2 = V_1 = V_2 = init_array
H_0 = Kinetic(T_1) + Kinetic(T_2) + Potential(V_1) + Potential(V_2)

#Eigenvalues
eigvals, eigvecs = eigsh(H_0, k = 2, which = 'SA')

#Wavefunction
u = eigvecs[:, n]

r = np.zeros(N)
for i in range(1, N):
    r[i] = (i * dr)
    
cusp = (1 - ((e**2 * dr * m_e) / (4 * np.pi))) #By Cusp Condition

psi = np.zeros([2, N])
for i in range(1, len(psi[0])):
    psi[0][i] = psi[1][i] = u[i] / r[i]
    
psi[0][0] = psi[0][1] / cusp
psi[1][0] = psi[1][1] / cusp

psi[0] = psi[0] * 1.0/np.sqrt(4 * np.pi)
psi[1] = psi[1] * 1.0/np.sqrt(4 * np.pi)

psi[0] = Norm(psi[0])
psi[1] = Norm(psi[1])

E_0 = Energy(psi[0]) + Energy(psi[1])
print("E_0:", E_0)
print("U_eff:", U_eff(psi, 0))
print("Total Energy:", E_0 + U_eff(psi, 0))
