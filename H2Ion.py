import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import random
from scipy.interpolate import make_interp_spline, BSpline
#from mpl_toolkits.mplot3d import Axes3D

#Constants
hbar = 1.0
kT = 1e-3
m_e = 0.511e6
e = 0.303
eps_0 = 1.0
dr = 3.012e-5
dtheta = 0.1
a_0 = 2.7e-4
Z = 1
N = 1000
E0 = -13.6

R = 0.51 * a_0
#---------------------------------------------------

def frange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step
    
def Overlap(arr_1, arr_2):
    integral = 0
    for radius in range(0, len(arr_1)):
        for angle in frange(0, np.pi, dtheta):
            #Units of Bohr Radii
            r_a = radius * dr
            r_b = np.sqrt(r_a**2 + R**2 - (2 * r_a * R * np.cos(angle)))
            #Switch to units of stepsize
            if round(r_b/dr) > len(arr_1):
                integral += 0
            elif round(r_b/dr) < len(arr_1):
                integral += arr_1[round(r_a/dr)] * arr_2[round(r_b/dr)] * (radius * dr)**2 * np.sin(angle)
    integral = integral * 2*np.pi * dr * dtheta
    return integral
        
def Norm(arr_1, arr_2):
    A = 1.0 / np.sqrt(2*(1 + Overlap(arr_1, arr_2)))
    return A
    
def Norm_single(array):
    J_array = np.zeros(len(array))
    for i in range(len(array)):
        J_array[i] = ((dr*i)**2) * array[i]
    A_single = 1.0 / np.sqrt(4*np.pi * dr * np.dot(J_array, array))
    norm_array = A_single * array
    return norm_array
    
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

def Energy_single(array):

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
    
def Energy_total(arr_1, arr_2):
    D = X = 0
    #Direct, Exchange Integrals
    for radius in range(0, len(arr_1)):
        for angle in frange(0, np.pi, dtheta):
            r_a = radius * dr
            r_b = np.sqrt(r_a**2 + R**2 - (2 * r_a * R * np.cos(angle)))
            if round(r_b/dr) > len(arr_1):
                D += 0
                X += 0
            elif round(r_b/dr) < len(arr_1):
                D += arr_1[radius]**2 * (radius * dr)**2 * np.sin(angle) * (1.0 / (r_b))
                X += arr_1[round(r_a/dr)] * arr_2[round(r_b/dr)] * (radius * dr) * np.sin(angle)
    D = D * 2*np.pi * dr * dtheta * a_0
    X = X * 2*np.pi * dr * dtheta * a_0
        
    expH = (Energy_single(arr_1) + Energy_single(arr_2))/2 - 2 * (Norm(arr_1, arr_2))**2 *(e**2 / (4 * np.pi * eps_0)) * 1.0/a_0 * (D + X) + (e**2 / (4 * np.pi * eps_0)) * (1.0 / R)
            
    return expH
    
#---------------------------------------------------------

r1 = np.linspace(0, 1000, 1000)
r2 = np.linspace(0, 1000, 1000)
R1, R2 = np.meshgrid(r1, r2)

psi = np.ones([2, N])
psi[0] = Norm_single(np.exp(-r1*dr/a_0))
psi[1] = Norm_single(np.exp(-r2*dr/a_0))

psi_composite = psi[0] + psi[1]
psi_composite = Norm(psi[0], psi[1]) * psi_composite

expH = Energy_total(psi[0], psi[1])


#VMC Alg:
cusp = (1 - ((e**2 * dr * m_e) / (4 * np.pi)))

r_list = []
E_list = []

for radii in range(25):
    for iterations in range(50):
        particle = random.randint(0, 1)
        pre_perturbed_psi_particle = psi[particle] * 1.0

        #Perturb Normalized Wavefunction
        k = random.randint(0, len(psi[particle]) - 1)
        h_shift = random.uniform(-0.05 * psi[particle][k], 0.05 * psi[particle][k])
        w_shift = random.uniform(10.0, 100.0)
        for i in range(len(psi[particle])):
            psi[particle][i] = psi[particle][i] - h_shift * np.exp(-(i - k)**2 / w_shift**2)
        
        #Boundary conditions for r -> +inf
        for i in range(int(len(psi[particle])*0.995), len(psi[particle])):
            psi[particle][i] = psi[np.abs(particle - 1)][i] = 0

        #Cusp Condition
        psi[particle][0] = psi[particle][1] / cusp

        #Normalize:
        psi[particle] = Norm_single(psi[particle])
        psi[np.abs(particle - 1)] = Norm_single(psi[np.abs(particle - 1)])


        #Energy:
        expH_prime = 0
        expH_prime = Energy_total(psi[0], psi[1])
        deltaE = expH_prime - expH
        

        #Metropolis Accept / Reject:
        if expH_prime < expH:
           #accept
           expH = expH_prime
        elif expH < expH_prime:
           P = np.exp(-deltaE / kT)
           u = random.random()
           if P < u:
               #reject
               psi[particle] = pre_perturbed_psi_particle * 1.0
           elif u < P:
               #accept
               expH = expH_prime
               
     
    r_list.append(R/a_0)
    E_list.append(expH)
    R += 0.5 * a_0
    
    
    psi = np.ones([2, N])
    psi[0] = Norm_single(np.exp(-r1*dr/a_0))
    psi[1] = Norm_single(np.exp(-r2*dr/a_0))
    expH = Energy_total(psi[0], psi[1])
    
r_list = np.array(r_list)
E_list = np.array(E_list)

rlist_new = np.linspace(r_list.min(), r_list.max(), 200)
spl = make_interp_spline(r_list, E_list, k=2)
Elist_smooth = spl(rlist_new)
plt.plot(rlist_new, Elist_smooth)
plt.title("$<H>_{min}$ vs. $R/a_0$")
plt.xlabel("$R/a_0$")
plt.ylabel("$<H>_{min}$")
plt.show()
    
    

