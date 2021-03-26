#Updated Eigensolver Algorithm in the |r, θ⟩ basis

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags
import matplotlib.pyplot as plt

#------------------------------------------------------------
#Constants

hbar = 1.0
e = 0.303
m_e = 0.511e6
eps_0 = 1
n = 0
Z = 1
a_0 = 2.7e-4
r_0 = 1e-3

N = 60
M = 60
dr = 5*a_0 / (N - 1)
dtheta = 2*np.pi / M


#------------------------------------------------------------
#Functions
    
def Lap():
    u = np.zeros([M*(N - 1) + 1, M*(N - 1) + 1])
     
    #Loop to set row (0, 0)
    u[0][0] = -4
    for i in range(1, M + 1):
        u[0][i] = 4/M
    
    #Loop for rows r = 1
    for i in range(1, M + 1):
        rad_1 = 1
        if i == 1:
            u[i][i], u[i][0], u[i][i + M], u[i][i + 1], u[i][i + M - 1] = -2.0 - (2.0 / (rad_1 * dtheta)**2), 1.0 - (1.0 / (2 * rad_1)), 1.0 + (1.0 / (2 * rad_1)), 1.0 / (rad_1 * dtheta)**2, 1.0 / (rad_1 * dtheta)**2
        elif  i > 1 and i < M:
            u[i][0], u[i][i], u[i][i + M], u[i][i - 1], u[i][i + 1] = 1.0 - (1.0 / (2 * rad_1)), -2.0 - (2.0 / (rad_1 * dtheta)**2), 1.0 + (1.0 / (2 * rad_1)), 1.0 / (rad_1 * dtheta)**2, 1.0 / (rad_1 * dtheta)**2
        elif i == M:
            u[i][0], u[i][i], u[i][i - 1], u[i][i + M], u[i][i - M + 1] = 1.0 - (1.0 / (2 * rad_1)), -2.0 - (2.0 / (rad_1 * dtheta)**2), 1.0 / (rad_1 * dtheta)**2, 1.0 + (1.0 / (2 * rad_1)), 1.0 / (rad_1 * dtheta)**2
     
          
    #Loop to set rows (2, 0) -> (N - 1, M)
    k = M + 1
    for rad in range(2, N - 1):
        for ang in range(0, M):
            if ang == 0:
                u[k][k], u[k][k - M], u[k][k + M], u[k][k + 1], u[k][k + M - 1] = -2.0 - (2.0 / (rad * dtheta)**2), 1.0 -  (1.0 / (2 * rad)), 1.0 +  (1.0 / (2 * rad)),  1.0 / (rad*dtheta)**2, 1.0 / (rad*dtheta)**2
            elif ang == M - 1:
                u[k][k], u[k][k - 1], u[k][k - M + 1], u[k][k + M], u[k][k - M] = -2.0 - (2.0 / (rad * dtheta)**2), 1.0 / (rad * dtheta)**2, 1.0 / (rad * dtheta)**2, 1.0 + (1.0 / (2 * rad)), 1.0 - (1.0 / (2 * rad))
            else:
                u[k][k], u[k][k - 1], u[k][k + 1], u[k][k + M], u[k][k - M] = -2.0 - (2.0 / (rad * dtheta)**2), 1.0 / (rad * dtheta)**2, 1.0 / (rad * dtheta)**2,  1.0 +  (1.0 / (2 * rad)),  1.0 -  (1.0 / (2 * rad))
            k += 1
            
    #Loop to set rows r = N - 1
    l = M * (N - 2) + 1
    rad = N - 1
    for ang in range(0, M):
        if ang == 0:
            u[l][l], u[l][l + 1], u[l][l + M - 1], u[l][l - M] = -2.0 - (2.0 / (rad * dtheta)**2), 1.0 / (rad * dtheta)**2, 1.0 / (rad * dtheta)**2, 1.0 - (1.0 / (2 * rad))
        elif ang == M - 1:
            u[l][l], u[l][l - 1], u[l][l - M + 1], u[l][l - M] = -2.0 - (2.0 / (rad * dtheta)**2), 1.0 / (rad * dtheta)**2, 1.0 / (rad * dtheta)**2, 1.0 - (1.0 / (2 * rad))
        else:
            u[l][l], u[l][l - 1], u[l][l + 1], u[l][l - M] = -2.0 - (2.0 / (rad * dtheta)**2), 1.0 / (rad * dtheta)**2, 1.0 / (rad * dtheta)**2, 1.0 - (1.0 / (2 * rad))
        l += 1
            
    return u
    

def Potential():

    v = np.zeros([M*(N - 1) + 1, M*(N - 1) + 1])

    c = 0
    for rad in range(1, N):
        for angle in range(0, M):
            v[c][c] = -1.0 / (rad * dr)
            c += 1
    v[0][0] = v[1][1]
    #v[0][0] = 2*v[1,1] - v[M + 1, M + 1]
    
    v = (Z*e**2 / (4*np.pi*eps_0)) * v
    return v

#------------------------------------------------------------

#Initialize Hamiltonian
T = (-hbar**2) / (2 * m_e * dr**2) * Lap()
V = Potential()
H = T + V


eigvals, eigvecs = eigsh(H, k = 1, which = 'SA')

print("Energy = ", eigvals[0])

fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")

u = eigvecs[:,0]
ax.scatter3D(0, 0, u[0], color = 'blue')

k = 1
for rad in range(1, N):
    for theta in range(M):
        ax.scatter3D(rad*dr, theta*dtheta, u[k], color = 'blue')
        k += 1

plt.title('Wavefunction')
ax.set_xlabel(r'Radius ($a_0$)')
ax.set_ylabel(r'$\theta$ (radians)')
ax.set_zlabel(r'$\psi(r, \theta)$')
plt.show()

#|v⟩ = [(0,0), (1, 0), (1, 1)...


