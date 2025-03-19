import numpy as np
import matplotlib.pyplot as plt

halonorm = 384748.91*0.1678331476*0.0748013263**(1/2)

Ay = 45000/halonorm
Az = 28692/halonorm
wxy = 1.86265
m = 0.0748013263
e = 0.054900489
e_m = e/m

## Define functions
def coordinates(t, Ay, Az, omega_xy0, m, order):
    theta = 0
    omega_2 = 0.0434986*Ay**2 - 0.1447905*Az**2
    omega = omega_xy0/(1 + m*omega_2)
    T = omega*t + theta

    # First order solution
    x1 = 0.341763*Ay*np.sin(T)
    y1 = Ay*np.cos(T)
    z1 = Az*np.sin(T)

    # Second order solution
    x2 = - (0.095884 * Ay**2 - 0.120121 * Az**2) * np.cos(2*T) - (0.205537 * Ay**2 + 0.268186 * Az**2)
    y2 = - (0.055296 * Ay**2 + 0.076511 * Az**2) * np.sin(2*T)
    z2 = Ay * Az * (0.1305819 * np.cos(2*T) + 0.3917467)

    # Third order solution
    epsilon = 218.32*np.pi/180
    omega_ = 280.459*np.pi/180
    phi = (1 - 3/4*m**2 - 225/32*m**3)*t + epsilon - omega_
    C1 = 0.02905497*Ay**2 + 0.00724195*Az**2
    x3 = (C1 * Ay * np.sin(T) - (0.030889 * Ay * Az**2 - 0.027808 * Ay**3) * np.sin(3*T) 
          + (e_m * Ay * (0.554904 * np.sin(phi - T) + 0.493213 * np.sin(phi + T))))
    y3 = ((0.001354 * Ay * Az**2 - 0.027574 * Ay**3) * np.cos(3*T)
         - (e_m * Ay * (1.905541 * np.cos(phi - T) - 1.210697 * np.cos(phi + T))))
    z3 = ((0.017581 * Az**3 - 0.043703 * Ay**2 * Az) * np.sin(3*T)
         + (e_m * Az * (1.760353 * np.sin(phi - T) + 1.020367 * np.sin(phi + T))))

    ## Total solution
    if order == 1:
        xn = m**0.5*x1
        yn = m**0.5*y1
        zn = m**0.5*z1
    if order == 2:
        xn = m**0.5*x1 + m*x2
        yn = m**0.5*y1 + m*y2
        zn = m**0.5*z1 + m*z2
    if order == 3:
        xn = m**0.5*x1 + m*x2 + m**1.5*x3
        yn = m**0.5*y1 + m*y2 + m**1.5*y3
        zn = m**0.5*z1 + m*z2 + m**1.5*z3
    return xn, yn, zn

## Change timestep and order here
t = np.linspace(0,84,2000)
order = 3

## Calculate coordinates
p = np.zeros((3,len(t)))
for i in range(len(t)):
    p[:,i] = coordinates(t[i], Ay, Az, wxy, m, order)

## Create plots
axs = plt.figure()
axs = plt.axes(projection='3d')
axs.plot3D(p[0,:], p[1,:], p[2,:])
axs.set_title(f'Halo orbit order {order}')
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_zlabel('Z')
axs.set_aspect('equal')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(10, 6))
axes[0].plot(p[0,:], p[1,:], color='blue')
axes[0].set_title('X vs Y')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
axes[0].set_aspect('equal')
axes[0].grid()
axes[1].plot(p[0,:], p[2,:], color='red')
axes[1].set_title('X vs Z')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Z')
axes[1].set_aspect('equal')
axes[1].grid()
axes[2].plot(p[1,:], p[2,:], color='green')
axes[2].set_title('Y vs Z')
axes[2].set_xlabel('Y')
axes[2].set_ylabel('Z')
axes[2].set_aspect('equal')
axes[2].grid()
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, figsize=(10, 6))
axes.plot(p[0,:], p[1,:], color='green')
axes.set_title(f'Halo orbit order {order}, Y vs Z')
axes.set_xlabel('Y')
axes.set_ylabel('Z')
axes.set_aspect('equal')
plt.tight_layout()
plt.grid()
plt.show()