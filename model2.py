import numpy as np
size = (4260, 2820)
cones = size[1] * size[0]
blue_cones = np.round(cones * 0.1)
green_cones = np.round(cones * 0.3)
red_cones = cones - blue_cones - green_cones
eye = []
bias = 1
for row in range(size[1]):
    eye.append([])
    for col in range(size[0]):
        blue_prob = blue_cones / cones
        green_prob = green_cones / cones
        red_prob = 1 - blue_prob - green_prob
        cone_probabilities = [red_prob, green_prob, blue_prob]
        for i in range(3):
            bias_factor = 1
            # find clusters based on number of steps to reach a cone
            if row > 0 and col > 0:
                proximity_1 = [eye[row - 1][col], eye[row][col - 1]].count(i)
                if proximity_1:
                    bias_factor += proximity_1
                    if row > 1 and col > 1 and col < size[0] - 1:
                        proximity_2 = [eye[row - 2][col], eye[row][col - 2], eye[row - 1][col - 1], eye[row - 1][col + 1]].count(i)
                        if proximity_2:
                            bias_factor += proximity_2
                            if row > 2 and col > 2 and col < size[0] - 2:
                                proximity_3 = [eye[row - 3][col], eye[row - 2][col - 1], eye[row - 1][col - 2], eye[row][col - 3], eye[row - 2][col + 1], eye[row - 1][col + 2]].count(i)
                                bias_factor += proximity_3
            new_bias = bias * bias_factor
            cone_probabilities[i] = np.log(1 + cone_probabilities[i] * new_bias) / np.log(1 + new_bias)
        cone_probabilities[2] *= np.sqrt(abs(row - size[1] / 2) ** 2 + abs(col - size[0] / 2) ** 2) / np.sqrt((size[1] / 2) ** 2 + (size[0] / 2) ** 2) * (5 / 3 - 1 / 3) + 1 / 3 # bias of blue to be on periphery
        cone_probabilities_sum = sum(cone_probabilities)
        for i in range(3):
            cone_probabilities[i] = cone_probabilities[i] / cone_probabilities_sum
        chosen_cone = np.random.choice([0, 1, 2], p=cone_probabilities) # 0 = red, 1 = green, 2 = blue
        if chosen_cone == 0:
            red_cones -= 1
            cones -= 1
        elif chosen_cone == 1:
            green_cones -= 1
            cones -= 1
        elif chosen_cone == 2:
            blue_cones -= 1
            cones -= 1
        eye[-1].append(chosen_cone)
np.save("eye.npy", eye)
