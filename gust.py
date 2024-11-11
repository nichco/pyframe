import numpy as np
import matplotlib.pyplot as plt

# Define the time array
t = np.linspace(0, 1, 100)

# Define the gust profile using the 1-cosine function
def gust_velocity(t, t0, t1, V_max):
    return np.where((t >= t0) & (t <= t1), V_max * 0.5 * (1 - np.cos(2*np.pi * (t - t0) / (t1 - t0))), 0)

# Parameters for the gust profile
t0 = 0
t1 = 1
V_max = 1

# Calculate the gust velocity
V_gust = gust_velocity(t, t0, t1, V_max)

# Plot the gust profile
plt.plot(t, V_gust, 'b')
plt.xlabel('Time, s')
plt.ylabel('Gust velocity, m/s')
plt.grid(True)
# plt.ylim(-1, 5)
# plt.xlim(0, 20)
plt.title('1-Cosine Gust Profile')
plt.show()
