# anti_swing_pd_crane.py
import numpy as np
import matplotlib.pyplot as plt

# params
m = 500.0        # kg
l = 12.0         # m
g = 10.0         # m/s^2
I = m * l**2     # moment of inertia

# design (choose wn_cl)
wn_cl = 1.5      # rad/s  <-- change to 1.0 or 2.0 to experiment
Kp = I * (wn_cl**2 - g/l)
Kd = 2 * I * wn_cl

# sim settings
dt = 0.001
T = 20.0
t = np.arange(0, T, dt)

# initial condition
theta = np.zeros_like(t)
omega = np.zeros_like(t)
theta[0] = np.deg2rad(17.0)   # 17 degrees initial swing

# actuator limits (saturation) - realistic motor torque limits
MAX_TORQUE = 5e5   # N*m (set per your motor capacity)
MIN_TORQUE = -MAX_TORQUE

def control(theta, omega):
    u = -Kp*theta - Kd*omega
    # saturation
    if u > MAX_TORQUE: u = MAX_TORQUE
    if u < MIN_TORQUE: u = MIN_TORQUE
    return u

# RK4 integration for theta'' = (u - m*g*l*sin(theta))/I
for i in range(len(t)-1):
    def accel(th, om):
        u = control(th, om)
        return (u - m*g*l*np.sin(th)) / I

    th = theta[i]
    om = omega[i]

    k1_om = accel(th, om)
    k1_th = om

    k2_om = accel(th + 0.5*dt*k1_th, om + 0.5*dt*k1_om)
    k2_th = om + 0.5*dt*k1_om

    k3_om = accel(th + 0.5*dt*k2_th, om + 0.5*dt*k2_om)
    k3_th = om + 0.5*dt*k2_om

    k4_om = accel(th + dt*k3_th, om + dt*k3_om)
    k4_th = om + dt*k3_om

    theta[i+1] = th + (dt/6.0)*(k1_th + 2*k2_th + 2*k3_th + k4_th)
    omega[i+1] = om + (dt/6.0)*(k1_om + 2*k2_om + 2*k3_om + k4_om)

# plot
plt.figure(figsize=(8,4))
plt.plot(t, np.rad2deg(theta))
plt.xlabel('Time (s)')
plt.ylabel('Swing angle (deg)')
plt.grid(True)
plt.title(f'PD anti-swing (wn_cl={wn_cl:.2f} rad/s)')
plt.show()
