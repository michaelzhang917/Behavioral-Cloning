import csv
import pandas as pd
import random
from keras.preprocessing.image import img_to_array, load_img, flip_axis, random_shift
import numpy as np
import pickle

def random_darken(image):
    """Given an image (from Image.open), randomly darken a part of it."""
    w, h = image.size

    # Make a random box.
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(x1, w), random.randint(y1, h)

    # Loop through every pixel of our box (*GASP*) and darken.
    for i in range(x1, x2):
        for j in range(y1, y2):
            new_value = tuple([int(x * 0.5) for x in image.getpixel((i, j))])
            image.putpixel((i, j), new_value)
    return image

def augment_image(image, angle):
    if random.random() < 0.5:
        image = random_darken(image)

    image = img_to_array(image)
    image = random_shift(image, 0, 0.2, 0, 1, 2)
    if random.random() < 0.5:
        image = flip_axis(image, 1)
        angle = - angle

    return image, angle

dataFolderPrefix = '../'
Folder = 'data/'
Filename = 'driving_log.csv'
drivingLogFile = dataFolderPrefix + Folder + Filename
offset = 0.2
data = pd.read_csv(drivingLogFile, header=0,
                   names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])

ROWS = 100
COLS = 100
speed_all = data.speed.values
X_all = []
y_all = []
for i in range(len(speed_all)):
    if float(speed_all[i]) < 20 : continue
    # Load center image
    imageFileName = dataFolderPrefix + Folder + data.center.values[i]
    image = load_img(imageFileName, target_size=(ROWS, COLS))
    angle = float(data.angle.values[i])
    image, angle = augment_image(image, angle)
    #image = image.astype(np.uint8)
    image = (image / 255. - .5).astype(np.float32)
    X_all.append(image)
    y_all.append(angle)
    print(imageFileName)
    # Load left image
    imageFileName = dataFolderPrefix + Folder + data.left.values[i].strip()
    image = load_img(imageFileName, target_size=(ROWS, COLS))
    angle = float(data.angle.values[i] + offset)
    image, angle = augment_image(image, angle)
    #image = image.astype(np.uint8)
    image = (image / 255. - .5).astype(np.float32)
    X_all.append(image)
    y_all.append(angle)
    print(imageFileName)
    # Load right image
    imageFileName = dataFolderPrefix + Folder + data.right.values[i].strip()
    image = load_img(imageFileName, target_size=(ROWS, COLS))
    angle = float(data.angle.values[i] - offset)
    image, angle = augment_image(image, angle)
    #image = image.astype(np.uint8)
    image = (image / 255. - .5).astype(np.float32)
    X_all.append(image)
    y_all.append(angle)
    print(imageFileName)

data = {"feature": X_all, "label": y_all}
with open('../preprocessedData/dataOFFSET=0.2' + str(ROWS) + 'x' + str(COLS) + '.p', 'wb') as f:
    pickle.dump(data, f)
