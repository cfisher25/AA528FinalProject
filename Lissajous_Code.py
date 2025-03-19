import numpy as np
import matplotlib.pyplot as plt

lissajousnorm = 384748.91*0.1678331476*0.0748013263

Ay = 3500/lissajousnorm
Az = 3500/lissajousnorm
wxy = 1.86265
wz = 1.78618
Ax = 0.343336*Ay
e = 0.054900489
m = 0.0748013263

def coordinates(t, Ax, Ay, Az, wxy, wz, order):
    epsilon = np.pi/2
    omega_ = 0
    phi = (1 - 3/4*m**2 - 225/32*m**3)*t + epsilon + omega_

    # First order solution
    x1 = Ax*np.sin(wxy*t)
    y1 = Ay*np.cos(wxy*t)
    z1 = Az*np.sin(wz*t)

    ## Second order solution
    theta1, theta2 = 0, 0
    T1 = wxy*t + theta1
    T2 = wz*t + theta2
    x2 = (0.554904 * (e/m) * Ay * np.sin(phi - T1) + 0.493213 * (e/m) * Ay * np.sin(phi + T1)
          - 0.09588405 * Ay**2 * np.cos(2*T1) + 0.128774 * Az**2 * np.cos(2*T2)
          - 0.268186 * Az**2 - 0.205537 * Ay**2)
    y2 = (-1.90554 * (e/m) * Ay * np.cos(phi - T1) + 1.210699 * (e/m) * Ay * np.cos(phi + T1)
          - 0.055296 * Ay**2 * np.sin(2*T1) - 0.08659705 * Az**2 * np.sin(2*T2))
    z2 = (1.052082 * (e/m) * Az * np.sin(phi + T2) + 1.856918 * (e/m) * Az * np.sin(phi - T2)
          + 0.4241194 * Ay * Az * np.cos(T2 - T1) + 0.1339910 * Ay * Az * np.cos(T2 + T1))
    
    ## Total solution
    if order == 1:
      x_n = m*x1
      y_n = m*y1
      z_n = m*z1
    if order == 2:
      x_n = m*x1 + m**2*x2
      y_n = m*y1 + m**2*y2
      z_n = m*z1 + m**2*z2
    return x_n, y_n, z_n

## Change timestep and order here
t = np.linspace(0,84,5000)
order = 2

## Calculate coordinates
p = np.zeros((3,len(t)))
for i in range(len(t)):
    p[:,i] = coordinates(t[i], Ax, Ay, Az, wxy, wz, order)

## Create Plots
axs = plt.figure()
axs = plt.axes(projection='3d')
axs.plot3D(p[0,:], p[1,:], p[2,:])
axs.set_xlabel('X')
axs.set_ylabel('Y')
axs.set_zlabel('Z')
axs.set_title(f'Lissajous orbit order {order}')
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
axes.plot(p[1,:], p[2,:], color='green')
# axes.set_title('Y vs Z')
axes.set_title(f'Lissajous orbit order {order}, Y vs Z')
axes.set_xlabel('Y')
axes.set_ylabel('Z')
axes.set_aspect('equal')
plt.tight_layout()
plt.grid()
plt.show()