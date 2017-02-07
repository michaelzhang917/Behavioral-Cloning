from keras.models import load_model, Sequential
from keras.layers import Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, Dense, Lambda
from keras.applications.vgg16 import  VGG16
import cv2
import numpy as np
import math
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint, EarlyStopping
import random, json
from keras import backend as K
from keras.preprocessing.image import load_img
from keras import optimizers
import pandas as pd


# Nvidia model based on paper "End to End Learning for Self-Driving Cars"
def get_nvidia_model(ROWS, COLS, CHANNELS, loadflag, filename='model.h5'):
    # option for loading the existing model
    if loadflag :
        return load_model(filename)
    weight_init = 'glorot_normal'

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0 - 0.5), input_shape=(ROWS, COLS, CHANNELS)))
    model.add(BatchNormalization(mode=2, axis=3, input_shape=(ROWS, COLS, CHANNELS)))
    model.add(Convolution2D(3, 3, 3, init=weight_init, border_mode='valid', activation='relu', input_shape=(ROWS, COLS, CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(9, 3, 3, init=weight_init, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(18, 3, 3, init=weight_init, border_mode='valid', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Convolution2D(32, 3, 3, init=weight_init, border_mode='valid',  activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(80, activation='relu', init=weight_init))

    model.add(Dense(15, activation='relu', init=weight_init))

    model.add(Dropout(0.25))
    model.add(Dense(1, init=weight_init, activation='linear'))

    return model


# VGG16 model
def get_vgg16_model(ROWS, COLS, CHANNELS, load):
    if load :
        return load_model('../models/vgg16/model.h5')
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=None, input_shape=(ROWS, COLS, CHANNELS))
    model = Sequential()
    model.add(vgg16)
    model.add(Flatten(input_shape=model.output_shape[1:]))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='linear'))

    return model


# The following functions are for jittering the data

# Adjust the brightness of the image
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1


# Transform the image up, down, left and right
def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 40 * np.random.uniform() - 40 / 2
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (image.shape[1], image.shape[0]))
    return image_tr, steer_ang


# Adding random shadow to the image
def add_random_shadow(image):
    top_y = image.shape[1]*np.random.uniform()
    top_x = 0
    bot_x = image.shape[0]
    bot_y = image.shape[1]*np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1
    if np.random.randint(2) == 1:
        random_bright = .5
        cond1 = shadow_mask == 1
        cond0 = shadow_mask == 0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image


# Resize the image
def preprocessImage(image, new_size):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/5):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size[0],new_size[1]), interpolation=cv2.INTER_AREA)
    return image


# The whole procedures for image augmentation
def imageAugument(img, sa, transRange, newSize):
    img, sa = trans_image(img, sa, transRange)
    img = augment_brightness_camera_images(img)
    img = add_random_shadow(img)
    img = preprocessImage(img, newSize)
    img = np.array(img)
    ind_flip = np.random.randint(2)
    if ind_flip == 0:
        img = cv2.flip(img, 1)
        sa = -sa
    return img, sa


# Batch data generator for training
def fit_gen(batch_size, fileNames, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(fileNames) - 1)
            sa = y[sample_index]
            img = load_img(fileNames[sample_index], target_size=(ROWS, COLS))
            img, sa = imageAugument(np.array(img).astype(np.uint8), sa, transRange, newSize)
            batch_X.append(img)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)

def generateFileList(Folder, offset):
    Filename = '/driving_log.csv'
    drivingLogFile = Folder + Filename
    data = pd.read_csv(drivingLogFile, header=0,
                       names=['center', 'left', 'right', 'angle', 'throttle', 'break', 'speed'])
    speed_all = data.speed.values
    X_all = []
    y_all = []
    for i in range(len(speed_all)):
        # Discarding image with speed less than 20
        if float(speed_all[i]) < 20: continue
        # Load center image
        imageFileName = data.center.values[i]
        angle = float(data.angle.values[i])
        X_all.append(imageFileName)
        y_all.append(angle)
        print(imageFileName)
        # Load left image
        imageFileName = data.left.values[i].strip()
        angle = float(data.angle.values[i] + offset)
        X_all.append(imageFileName)
        y_all.append(angle)
        print(imageFileName)
        # Load right image
        imageFileName = data.right.values[i].strip()
        angle = float(data.angle.values[i] - offset)
        X_all.append(imageFileName)
        y_all.append(angle)
        print(imageFileName)

    data = {"fileNames": X_all, "label": y_all}
    with open(Folder + 'OFFSET=' +str(offset) + '.p', 'wb') as f:
        pickle.dump(data, f)


# Dimension of the original image
ROWS = 160
COLS = 320

# Parameters for image augmentation
transRange = 100
newRow = 128
newCol = 128
newSize = (newCol, newRow)
CHANNELS = 3
modelName = 'nvidia'
fileDate = '20170204'
load = False
prev_nb_epoch = 0

# Load the model

model = get_nvidia_model(newRow, newCol, CHANNELS, load=load,
                         filename='model.h5')


model.compile(loss='mse',
              optimizer=optimizers.Adam(lr=1e-4))

model.summary()

# Generate file list for training
offset = 0.2
Folder = 'data20170204'
generateFileList(Folder, offset)

with open(Folder + 'OFFSET=' +str(offset) + '.p', 'rb') as f:
    data = pickle.load(f)

fileNames_all = data['fileNames']
sa_all = data['label']

# Parameters for training
nb_epoch = 20
batch_size = 128

# Creating Validation Data; Using 20% training data as valiadtion data
fileNames_train, fileNames_test, sa_train, sa_test = train_test_split(
    fileNames_all, sa_all, test_size=0.20, random_state=23)

# Load all the validation data
X_valid, y_valid = [], []
for i in range(len(sa_test)):
    sa = sa_test[i]
    img = load_img(fileNames_test[i], target_size=(ROWS, COLS))
    img, sa = imageAugument(np.array(img).astype(np.uint8), sa, transRange, newSize)
    X_valid.append(img)
    y_valid.append(sa)


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
save_weights = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True)

# Train the model
model.fit_generator(fit_gen(batch_size, fileNames_train, sa_train),
                    samples_per_epoch=len(fileNames_train), nb_epoch=nb_epoch,
                    validation_data=(np.array(X_valid), np.array(y_valid)), callbacks=[save_weights, early_stopping])

# Model prediction on the validation set
preds = model.predict(np.array(X_valid), verbose=1)

# MSE and RMSE on the validation set
print( "Test MSE: {}".format(mean_squared_error(np.array(y_valid), preds)))
print( "Test RMSE: {}".format(np.sqrt(mean_squared_error(np.array(y_valid), preds))))

K.clear_session()
