from Models import *
from imageAugument import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import numpy as np
import random, json
from keras import backend as K
from keras.preprocessing.image import img_to_array, load_img
import cv2

ROWS = 160
COLS = 320
transRange = 100
newSize = (64, 64)
CHANNELS = 3
modelName = 'nvidia'
load = False

if modelName is 'nvidia':
    model = get_nvidia_model(ROWS, COLS, CHANNELS, load=load)
elif modelName is 'vgg16':
    model = get_vgg16_model(ROWS, COLS, CHANNELS, load=load)


model.compile(loss='mse',
              optimizer=optimizers.Adam(lr=1e-4))
              #metrics=['accuracy'])
model.summary()


def fit_gen(batch_size, fileNames, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(fileNames) - 1)
            sa = y[sample_index]
            img = load_img(fileNames[sample_index], target_size=(ROWS, COLS))
            img, sa = imageAugument(np.array(img).astype(np.uint8), sa, transRange, newSize)
            if modelName is 'vgg16':
               # X[sample_index].astype(np.float32)
                img[:,:,0] -= 103.939
                img[:,:,1] -= 116.779
                img[:,:,2] -= 123.68
            #img = (img / 255. - .5).astype(np.float32)
            batch_X.append(img)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)


offset = 0.2
with open('../preprocessedData/dataOFFSET=' +str(offset) + '.p', 'rb') as f:
    data = pickle.load(f)

fileNames_all = data['fileNames']
sa_all = data['label']

nb_epoch = 5
batch_size = 64

### Creating Validation Data
fileNames_train, fileNames_test, sa_train, sa_test = train_test_split(
    fileNames_all, sa_all, test_size=0.20, random_state=23)

X_test, y_test = [], []

for i in range(len(sa_test)):
    sa = sa_test[i]
    img = load_img(fileNames_test[i], target_size=(ROWS, COLS))
    img, sa = imageAugument(np.array(img).astype(np.uint8), sa, transRange, newSize)
    if modelName is 'vgg16':
        # X[sample_index].astype(np.float32)
        img[:, :, 0] -= 103.939
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 123.68
    # img = (img / 255. - .5).astype(np.float32)
    X_test.append(img)
    y_test.append(sa)


# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
modelFile = '../models/' + modelName + '/model' + 'OFFSET=' + str(offset) + 'size' + str(newSize[0]) + 'x' + str(newSize[1])
save_weights = ModelCheckpoint(modelFile + '.h5', monitor='val_loss', save_best_only=True)

model.fit_generator(fit_gen(batch_size, fileNames_train, sa_train),
        samples_per_epoch=len(fileNames_train), nb_epoch=nb_epoch,
        validation_data=(np.array(X_test), np.array(y_test)), callbacks=[save_weights, early_stopping])


preds = model.predict(np.array(X_test), verbose=1)

print( "Test MSE: {}".format(mean_squared_error(np.array(y_test), preds)))
print( "Test RMSE: {}".format(np.sqrt(mean_squared_error(np.array(y_test), preds))))

with open(modelFile + '.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

K.clear_session()