import numpy as np
import matplotlib.pyplot as plt
x = np.array([1, 2, 3, 4, 5])
y = np.load("mutual_information.npy")
plt.plot(x, y)
plt.xlabel("dimension")
plt.ylabel("mutual information")
plt.xticks(range(1, 6))
plt.legend()
plt.show()
x2 = np.log([1, 2, 3, 4, 5])
y2 = np.log(y)
p = np.polyfit(x2, y2, 1)
