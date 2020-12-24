import numpy as np
import random
import matplotlib.pyplot as plt

#Constants
hbar = 1.0
kT = 1.0e-3
m_e = 0.5e6
e = 0.303
eps_0 = 1.0
Z = 2

a_0 = 2.7e-4
N = int(4e2)
dr = 5*a_0 / N

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
    

#-----------------------------------------------
#Initialize Wavefunction
psi = np.ones([2, N])
psi[0] = np.fromiter((np.exp(-Z*i*dr/a_0) for i in range(len(psi[0]))), float)
psi[1] = np.fromiter((np.exp(-Z*i*dr/a_0) for i in range(len(psi[1]))), float)

#Normalize Wavefunction
psi[0] = Norm(psi[0])
psi[1] = Norm(psi[1])

#Hamiltonian; Initial Energy
E = Energy(psi[0]) + Energy(psi[1]) + U_eff(psi, 1)
print(Energy(psi[0]) + Energy(psi[1]))

#Metropolis Algorithm:

cusp = (1 - ((e**2 * dr * m_e) / (4 * np.pi)))
l = 0

En_list = []
for iter in range(int(1e4)):
#while True:
    l += 1
    particle = random.randint(0, 1)
    pre_perturbed_psi_particle = psi[particle] * 1.0
    
    
    #Perturb Normalized Wavefunction
    k = random.randint(0, len(psi[particle]) - 1)
    h_shift = random.uniform(-0.02 * psi[particle][k], 0.02 * psi[particle][k])
    w_shift = random.uniform(1.0, 100.0)
    for i in range(len(psi[particle])):
        psi[particle][i] = psi[particle][i] - h_shift * np.exp(-(i - k)**2 / w_shift**2)
        
    #Boundary conditions for r -> +inf
    for i in range(int(len(psi[particle])*0.995), len(psi[particle])):
        psi[particle][i] = psi[np.abs(particle - 1)][i] = 0
    
    
    #Cusp Condition
    psi[particle][0] = psi[particle][1] / cusp
    
    #Normalize
    psi_prime_perturbed = psi_prime_unperturbed = np.zeros(N)
    psi_prime_perturbed = Norm(psi[particle])
    psi_prime_unperturbed = Norm(psi[np.abs(particle - 1)])
    
    psi_prime_composite = np.zeros([2, N])
    psi_prime_composite[particle] = psi_prime_perturbed * 1.0
    psi_prime_composite[np.abs(particle - 1)] = psi_prime_unperturbed * 1.0
    
    #Energy
    E_prime = Energy(psi_prime_perturbed) + Energy(psi_prime_unperturbed) + U_eff(psi_prime_composite, particle)
    delta_E = E_prime - E
        
   #Metropolis Accept/Reject:
    if E_prime < E:
       #accept
       E = E_prime
       psi[particle] = psi_prime_composite[particle] * 1.0
    elif E < E_prime:
       P = np.exp(-delta_E / kT)
       u = random.random()
       if P < u:
           #reject
           psi[particle] = pre_perturbed_psi_particle * 1.0
       elif u < P:
           #accept
           E = E_prime
           psi[particle] = psi_prime_composite[particle] * 1.0
        
    
    #Animation
    '''scaling = np.amax(psi[0]) / len(psi[0])
    if l % 100 == 0:
        for q in range(len(psi[0]) - 1):
            print("l3", q, psi[0][q] / scaling, 0, q + 1, psi[0][q + 1] / scaling, 0)
            #print("l3", q, psi[1][q] / scaling, 0, q + 1, psi[1][q + 1] / scaling, 50)
        print("F")'''
        
        
    '''if l > 1500:
            En_list.append(E)
                
En_list = np.array(En_list)
avgE = np.average(En_list)
print(avgE)'''



'''fig, (ax1, ax2) = plt.subplots(2)
ax1.plot(psi[0])
ax1.set_title(r'$\psi_1$')
ax2.plot(psi[1])
ax2.set_title(r'$\psi_2$')
fig.subplots_adjust(hspace = 0.5)
plt.show()'''

