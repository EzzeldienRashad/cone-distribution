import numpy as np
size = (4260, 2820)
eye = []
for row in range(size[1]):
    eye.append([])
    for col in range(size[0]):
        eye[-1].append(np.random.choice([0, 1, 2], p=[0.3, 0.3, 0.4]))
np.save("eye.npy", eye)
