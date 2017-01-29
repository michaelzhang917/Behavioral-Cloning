from keras.models import load_model, Sequential
from keras.layers import Input, Dropout, Flatten, Convolution2D, MaxPooling2D, BatchNormalization, Dense, Lambda
from keras.applications.vgg16 import  VGG16

def get_nvidia_model(ROWS, COLS, CHANNELS, load):
    """Define hyperparameters and compile model"""
    if load: return load_model('../models/nvidia/model.h5')
    #lr = 0.0001
    weight_init='glorot_normal'
    #opt = optimizers.RMSprop(lr)
    #loss = 'mean_squared_error'

    model = Sequential()
    model.add(Lambda(lambda x: (x / 255.0 - 0.5), input_shape=(ROWS, COLS, CHANNELS)))
    #model.add(BatchNormalization(mode=2, axis=3, input_shape=(ROWS, COLS, CHANNELS)))
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

    #model.compile(optimizer=opt, loss=loss)

    return model


def get_vgg16_model(ROWS, COLS, CHANNELS, load):
    if load: return load_model('../models/vgg16/model.h5')

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

