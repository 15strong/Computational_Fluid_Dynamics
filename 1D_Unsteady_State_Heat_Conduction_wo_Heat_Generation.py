import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
rho = 7750.0
cp = 500.0
k = 16.2
L = 1.0
imax = 12
T0 = 30.0
T_wb = 0.0
T_eb = 100.0
Q_vol_gen = 0.0
epsilon_st = 0.0001

# --- Geometric & Time Step ---
alpha = k / (rho * cp)
Dx = L / (imax - 2)
Dt = 0.99 * 0.5 * (Dx ** 2 / alpha)
DTc = T_eb - T_wb
Q_gen = Q_vol_gen * Dx

# --- Initial & Boundary Conditions ---
T = np.zeros(imax)
T[1:-1] = T0  # interior
T[0] = T_wb
T[-1] = T_eb

# --- Time Loop ---
n = 0
unsteadiness_nd = 1.0

while unsteadiness_nd >= epsilon_st:
    n += 1
    T_old = T.copy()

    qx_old = np.zeros(imax)
    Q_cond_old = np.zeros(imax)

    for i in range(1, imax - 1):
        if i == 1 or i == imax - 2:
            qx_old[i] = -k * (T_old[i + 1] - T_old[i]) / (Dx / 2.0)
        else:
            qx_old[i] = -k * (T_old[i + 1] - T_old[i]) / Dx

    for i in range(2, imax - 1):
        Q_cond_old[i] = qx_old[i - 1] - qx_old[i]
        T[i] = T_old[i] + (Dt / (rho * cp * Dx)) * (Q_cond_old[i] + Q_gen)

    unsteadiness = np.max(np.abs(T - T_old)) / Dt
    unsteadiness_nd = unsteadiness * L ** 2 / (alpha * DTc)


# --- Plot the Result ---
m=np.array([0,1,3,5,7,9,11,13,15,17,19,20])
x = m*(Dx/2)
plt.plot(x, T, marker='o')
plt.xlabel("Length (m)")
plt.ylabel("Temperature (°C)")
plt.title("1D Unsteady Heat Conduction (Steady State)")
plt.grid(True)
plt.show()
