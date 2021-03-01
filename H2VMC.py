import numpy as np
import matplotlib.pyplot as plt
import random
import time

#Constants
hbar = 1.0
kT = 1e-2
m_e = 0.511e6
e = 0.303
eps_0 = 1.0
a_0 = 2.7e-4
Z = 1
N = 1000
dr = 3.012e-5
dtheta = 0.1
E0 = -13.6

cusp = (1 - ((e**2 * dr * m_e) / (4 * np.pi)))

R = 1.0 * a_0

#-------------------------

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
        
def Norm(array):
    A = 1.0 / np.sqrt(2*(1 + Overlap(array[0], array[1]) * Overlap(array[2], array[3])))
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
    
def Direct_int(arr_1):
    D = 0
    for radius in range(0, len(arr_1)):
        for angle in frange(0, np.pi, dtheta):
            D += arr_1[radius]**2 * (radius * dr) * np.sin(angle)
    D = D * 2*np.pi * dr * dtheta
    return D
    
def Coulomb_int(arr_1):
    D = 0
    for radius in range(0, len(arr_1)):
        for angle in frange(0, np.pi, dtheta):
            r_a = radius * dr
            r_b = np.sqrt(r_a**2 + R**2 - (2 * r_a * R * np.cos(angle)))
            if round(r_b/dr) > len(arr_1):
                D += 0
            elif round(r_b/dr) < len(arr_1):
                D += arr_1[radius]**2 * (radius * dr)**2 * np.sin(angle) * (1.0 / (r_b))
    D = D * 2*np.pi * dr * dtheta
    return D
    
def Exchange_int(array, radius_of_interest, radius_other):
    X = 0
    for radius in range(0, len(array[0])):
        for angle in frange(0, np.pi, dtheta):
            r_a = radius * dr
            r_b = np.sqrt(r_a**2 + R**2 - (2 * r_a * R * np.cos(angle)))
            if round(r_b/dr) > len(array[0]):
                X += 0
            elif round(r_b/dr) < len(array[0]):
                X += array[radius_of_interest][round(r_a/dr)] * array[radius_other][round(r_b/dr)] * (radius * dr) * np.sin(angle)
    X = X * 2*np.pi * dr * dtheta
    return X
    
def Phi(array):
    phi = np.zeros(N)
    dr_1 = dr
    dr_2 = dr + 1.1e-6
    
    for rad_2 in range(0, N):
        r_2 = rad_2 * dr_2
        int = 0
        if rad_2 == 0:
            for rad_1 in range(0, N):
                for theta in frange(0, np.pi, dtheta):
                    int += array[rad_1]**2 * (rad_1 * dr) * np.sin(theta)
        elif rad_2 > 0:
            for rad_1 in range(0, N):
                for theta in frange(0, np.pi, dtheta):
                    r_1 = rad_1 * dr_1
                    int += array[rad_1]**2 * (1.0 / np.sqrt(r_1**2 + r_2**2 - 2*r_1*r_2*np.cos(theta))) * r_1**2 * np.sin(theta)
        int = 2 * np.pi * dr * dtheta * int
        phi[rad_2] = int
    return phi
    
def Phi_prime(arr_1, arr_2):
    phi_prime = np.zeros(N)
    dr_1 = dr
    dr_2 = dr + 1.1e-6
    
    for rad_2 in range(0, N):
        r_2 = rad_2 * dr_2
        int = 0
        if rad_2 == 0:
            for rad_1 in range(0, N):
                for theta in frange(0, np.pi, dtheta):
                    r_1 = rad_1 * dr_1
                    r_1_prime = np.sqrt(r_1**2 + R**2 - 2 * r_1 * R * np.cos(theta))
                    if round(r_1_prime / dr) > N:
                        int += 0
                    elif round(r_1_prime / dr) < N:
                        int += arr_1[rad_1] * arr_2[round(r_1_prime/dr)] * (rad_1 * dr) * np.sin(theta)
        elif rad_2 > 0:
            for rad_1 in range(0, N):
                for theta in frange(0, np.pi, dtheta):
                    r_1 = rad_1 * dr_1
                    r_1_prime = np.sqrt(r_1**2 + R**2 - 2 * r_1 * R * np.cos(theta))
                    if round(r_1_prime / dr) > N:
                        int += 0
                    elif round(r_1_prime / dr) < N:
                        int += arr_1[rad_1] * arr_2[round(r_1_prime / dr)] * (1.0 / np.sqrt(r_1**2 + r_2**2 - 2*r_1*r_2*np.cos(theta))) * r_1**2 * np.sin(theta)
        int = 2 * np.pi * dr * dtheta * int
        phi_prime[rad_2] = int

    return phi_prime
    
def X2_int(array):
    arr_1, arr_2, arr_3, arr_4 = array[0], array[1], array[2], array[3]
    X2 = 0
    arr_5 = Phi_prime(arr_1, arr_2) * 1.0
    
    for radius in range(0, N):
        for theta in frange(0, np.pi, dtheta):
            r_2 = radius * dr
            r_2prime = np.sqrt(r_2**2 + R**2 - 2*r_2*R*np.cos(theta))
            if round(r_2prime/dr) > N:
                X2 += 0
            elif round(r_2prime/dr) < N:
                X2 += arr_3[radius] * arr_4[round(r_2prime/dr)] * arr_5[radius] * (radius*dr)**2 * np.sin(theta)
    X2 = 2*np.pi * dr * dtheta * X2

    return X2
    
def D2_int(arr_1, arr_2):
    #arr_1 = psi[3], arr_2 = psi[0]
    D2 = 0
    arr_3 = Phi(arr_2) * 1.0
    
    for radius in range(0, N):
        for theta in frange(0, np.pi, dtheta):
            r_2 = radius * dr
            r_2prime = np.sqrt(r_2**2 + R**2 - 2*r_2*R*np.cos(theta))
            if round(r_2prime/dr) > N:
                D2 += 0
            elif round(r_2prime/dr) < N:
                D2 += arr_1[round(r_2prime / dr)]**2 * arr_3[radius] * (radius*dr)**2 * np.sin(theta)
    D2 = 2*np.pi * dr * dtheta * D2
    return D2

def Energy_total(array):

    expT_1 = Norm(array)**2 * ((Energy_single(array[0]) + Energy_single(array[1])) * (1 + Overlap(array[0], array[1]) * Overlap(array[2], array[3])) + ((e**2)/(4*np.pi * eps_0))*(Direct_int(array[0]) + Exchange_int(array, 1, 0) * Overlap(array[2], array[3]) + Exchange_int(array, 0, 1) * Overlap(array[2], array[3]) + Direct_int(array[1])))
    
    expT_2 = Norm(array)**2 * ((Energy_single(array[2]) + Energy_single(array[3])) * (1 + Overlap(array[0], array[1]) * Overlap(array[2], array[3])) + ((e**2)/(4*np.pi * eps_0))*(Direct_int(array[2]) + Exchange_int(array, 3, 2) * Overlap(array[0], array[1]) + Exchange_int(array, 2, 3) * Overlap(array[0], array[1]) + Direct_int(array[3])))
    
    expV_1 = ((-e**2)/(4*np.pi * eps_0)) * Norm(array)**2 * (Direct_int(array[0]) + 2*Exchange_int(array, 0, 1)*Overlap(array[2], array[3]) + Coulomb_int(array[1]))
    
    expV_1prime = ((-e**2)/(4*np.pi * eps_0)) * Norm(array)**2 * (Coulomb_int(array[0]) + 2*Exchange_int(array, 1, 0)*Overlap(array[2], array[3]) + Direct_int(array[1]))
    
    expV_2 = ((-e**2)/(4*np.pi * eps_0)) * Norm(array)**2 * (Direct_int(array[2]) + 2*Exchange_int(array, 3, 2)*Overlap(array[0], array[1]) + Coulomb_int(array[3]))
    
    expV_2prime = ((-e**2)/(4*np.pi * eps_0)) * Norm(array)**2 * (Coulomb_int(array[2]) + 2*Exchange_int(array, 3, 2)*Overlap(array[0], array[1]) + Direct_int(array[3]))
    
    V_ee = ((e**2)/(4*np.pi * eps_0)) * Norm(array)**2 * 2*(D2_int(array[3], array[0]) + X2_int(array))
    
    V_pp = (e**2 / (4*np.pi*eps_0 * R))
    
    expH = np.sum([expT_1, expT_2,
                    expV_1, expV_2, expV_1prime, expV_2prime,
                    V_pp, V_ee])
    
    return expH

    
#---------------------

#Initialize Wavefunction
psi = np.ones([4, N])
psi[0] = psi[1] = psi[2] = psi[3] = Norm_single(np.fromiter((np.exp(-Z*i*dr/a_0) for i in range(len(psi[0]))), float))


expH = Energy_total(psi)
print(expH)

#VMC Alg

for iterations in range(50):
    particle = random.randint(0, 3)
    pre_perturbed_psi_particle = psi[particle] * 1.0

    #Perturb Normalized Wavefunction
    k = random.randint(0, len(psi[particle]) - 1)
    h_shift = random.uniform(-0.5 * psi[particle][k], 0.5 * psi[particle][k])
    w_shift = random.uniform(10.0, 50.0)
    for i in range(len(psi[particle])):
        psi[particle][i] = psi[particle][i] - h_shift * np.exp(-(i - k)**2 / w_shift**2)
    
    #Boundary conditions for r -> +inf
    for i in range(int(len(psi[particle])*0.995), len(psi[particle])):
        psi[particle][i] = 0

    #Cusp Condition (BC for r -> 0)
    psi[particle][0] = psi[particle][1] / cusp

    #Normalize:
    psi[particle] = Norm_single(psi[particle])

    #Energy:
    expH_prime = Energy_total(psi)
    deltaE = expH_prime - expH
    
    print(expH, expH_prime)

    #Metropolis Accept / Reject:
    if expH_prime < expH:
       #accept
       expH = expH_prime
       print(iterations, "accept type 1")
    elif expH < expH_prime:
       P = np.exp(-deltaE / kT)
       u = random.random()
       if P < u:
           #reject
           print(iterations, "reject", P, u)
           psi[particle] = pre_perturbed_psi_particle * 1.0
       elif u < P:
           #accept
           expH = expH_prime
           print(iterations, "accept type 2")
           
    
