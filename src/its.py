# Plot iterations

import matplotlib.pyplot as plt
import numpy as np

# Iterations for ex2 for various omega
# omega = 0.75
grid = np.array([3, 5, 7, 9, 11, 13, 15])

res_0p75 = np.array([36, 72, 116, 167, 224, 285, 350])
res_1p0 = np.array([20, 43, 72, 104, 140, 180, 222])
res_1p25 = np.array([12, 24, 42, 63, 87, 112, 140])
res_1p5 = np.array([22, 20, 21, 30, 46, 62, 79])
res_1p75 = np.array([52, 48, 48, 47, 44, 43, 44])

plt.plot(grid, res_0p75)
plt.plot(grid, res_1p0)
plt.plot(grid, res_1p25)
plt.plot(grid, res_1p5)
plt.plot(grid, res_1p75)

plt.legend([0.75, 1.0, 1.25, 1.5, 1.75])

plt.title("Iterations for SOR solver for various relaxation constants")
plt.xlabel("Grid size (N)")
plt.ylabel("Iterations")
plt.show()
