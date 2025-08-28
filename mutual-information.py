import numpy as np
x_dict = np.load("x_dict.npy", allow_pickle=True)[()]
y_dict = np.load("y_dict.npy", allow_pickle=True)[()]
mutual_infos = []
avg_xxT = []
avg_yyT = []
avg_xyT = []
avg_x = [[], [], [], [], []]
avg_y = [[], [], [], [], []]
for i in range(5):
    dimension = i + 1
    avg_xxT.append(np.zeros((dimension ** 2, dimension ** 2)))
    avg_yyT.append(np.zeros((dimension ** 2, dimension ** 2)))
    avg_xyT.append(np.zeros((dimension ** 2, dimension ** 2)))
for filename in x_dict.keys():
    for i in range(len(x_dict[filename])):
        for j in range(len(x_dict[filename][i])):
            avg_xxT[i] += x_dict[filename][i][j].astype("int")@x_dict[filename][i][j].astype("int").T
            avg_yyT[i] += y_dict[filename][i][j].astype("int")@y_dict[filename][i][j].astype("int").T
            avg_xyT[i] += x_dict[filename][i][j].astype("int")@y_dict[filename][i][j].astype("int").T
for i in range(5):
    dimension = i + 1
    avg_xxT[i] /= len(x_dict) * 5
    avg_yyT[i] /= len(y_dict) * 5
    avg_xyT[i] /= len(x_dict) * 5
    avg_x_dim_arr = np.zeros(dimension ** 2)[np.newaxis].T
    avg_y_dim_arr = np.zeros(dimension ** 2)[np.newaxis].T
    for j in range(5):
        for filename in x_dict.keys():
            avg_x_dim_arr += np.array(x_dict[filename][i][j])
            avg_y_dim_arr += np.array(y_dict[filename][i][j])
    avg_x[i] = avg_x_dim_arr / len(x_dict) / 5
    avg_y[i] = avg_y_dim_arr / len(y_dict) / 5
for i in range(5):
    dimension = i + 1
    sigma_x = avg_xxT[i] - avg_x[i]@avg_x[i].T
    sigma_y = avg_yyT[i] - avg_y[i]@avg_y[i].T
    sigma_xy = avg_xyT[i] - avg_x[i]@avg_y[i].T
    sigma = np.zeros([dimension ** 2 * 2, dimension ** 2 * 2])
    sigma[:dimension ** 2, :dimension ** 2] = sigma_x
    sigma[:dimension ** 2, dimension ** 2:] = sigma_xy
    sigma[dimension ** 2:, :dimension ** 2] = sigma_xy.T
    sigma[dimension ** 2:, dimension ** 2:] = sigma_y
    mutual_infos.append(1/2 * np.log(np.linalg.det(sigma_x) * np.linalg.det(sigma_y) / np.linalg.det(sigma)))
np.save("mutual_information.npy", mutual_infos)
