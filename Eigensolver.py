import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

#---------------------------------

#Constants
N = int(5e3)
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
    
def Lap(array):
    Lap_array = np.zeros(len(array))
    darray = Diff(array)
    rsqr_darray = np.fromiter((darray[i] * (dr*i)**2 for i in range(len(darray))), float)
    drsqr_darray = Diff(rsqr_darray)
    for i in range(1, len(array)):
        Lap_array[i] = drsqr_darray[i] / ((dr*i)**2)
    return Lap_array
    
def Norm(array):
    J_array = np.zeros(len(array))
    for i in range(len(array)):
        J_array[i] = ((dr*i)**2) * array[i]
    A = 1.0 / np.sqrt(4*np.pi * dr * np.dot(J_array, array))
    norm_array = A * array
    return norm_array
    
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
    array = (e**2 / (4*np.pi*eps_0)) * array
    return array
    
#----------------------------------

#Kinetic Energy
T = np.zeros([N,N], dtype = float)
T = Kinetic(T)

#Potential Energy
V = np.zeros([N, N], dtype = float)
V = Potential(V)

#Hamiltonian
H = T + V

#Eigenvalues
eigvals, eigvecs = eigsh(H, k = N - 1, which = 'SA')

#Wavefunction
u = eigvecs[:, n]

r = np.zeros(N)
for i in range(1, N):
    r[i] = (i * dr)
    
psi = np.zeros(N)
for i in range(1, N):
    psi[i] = u[i] / r[i]
psi[0] = psi[1] / (1 - ((e**2 * dr * m_e) / (4 * np.pi))) #By Cusp Condition
psi = psi * (1.0/np.sqrt(4*np.pi)) #Angular Wavefunction
npsi = Norm(psi)
print(Energy(npsi))


#----------------------------------

#Plot
#for k in range(len(u)):
    #print(k, psi[k])
    
    





