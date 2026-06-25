import matplotlib.pyplot as plt
import numpy as np
def brownian_motion(n, dt):
    increments = np.random.normal(0, np.sqrt(dt), size=n)
    path = np.cumsum(increments)
    return path
n = 1000
dt = 0.01
path = brownian_motion(n, dt)
plt.plot(path)
plt.show()