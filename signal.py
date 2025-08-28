from PIL import Image
import numpy as np
import os
x_dict = {}
y_dict = {}
for file in os.listdir(os.fsencode("images")):
    filename = os.fsdecode(file)
    x_dict[filename] = [[], [], [], [], []]
    y_dict[filename] = [[], [], [], [], []]
    img = Image.open("images/" + filename)
    img = img.crop((0, 0, 4260, 2820))
    img_array = np.array(img).astype("uint8")
    eye = np.load("eye.npy").astype("uint8")
    intensities = []
    for row in range(len(img_array)):
        intensities.append([])
        for col in range(len(img_array[row])):
            intensities[-1].append(img_array[row][col][eye[row][col]])
    intensities = np.array(intensities).astype("uint8")
    img_array_grayscale = np.array(img.convert("L")).astype("uint8")
    for i in range(5):
        brightness = np.random.poisson(lam=intensities).astype("uint8")
        for j in range(5):
            new_dimension = j + 1
            img_array_grayscale_small = img_array_grayscale.reshape([new_dimension, img_array_grayscale.shape[0] // new_dimension, new_dimension, img_array_grayscale.shape[1] // new_dimension]).mean(3).mean(1).astype("uint8")
            brightness_small = brightness.reshape([new_dimension, brightness.shape[0] // new_dimension, new_dimension, brightness.shape[1] // new_dimension]).mean(3).mean(1).astype("uint8")
            x_dict[filename][j].append(img_array_grayscale_small.ravel()[np.newaxis].T)
            y_dict[filename][j].append(brightness_small.ravel()[np.newaxis].T)
np.save("x_dict.npy", x_dict)
np.save("y_dict.npy", y_dict)
