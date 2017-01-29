from Models import *
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
import numpy as np
import random, json
from keras import backend as K

ROWS = 100
COLS = 100
CHANNELS = 3
modelName = 'vgg16'
load = True

if modelName is 'nvidia':
    model = get_nvidia_model(ROWS, COLS, CHANNELS, load=load)
elif modelName is 'vgg16':
    model = get_vgg16_model(ROWS, COLS, CHANNELS, load=load)


model.compile(loss='mse',
              optimizer=optimizers.Adam(lr=1e-4))
              #metrics=['accuracy'])
model.summary()

def fit_gen(batch_size, X, y):
    """Generate batches of training data forever."""
    while 1:
        batch_X, batch_y = [], []
        for i in range(batch_size):
            sample_index = random.randint(0, len(X) - 1)
            sa = y[sample_index]
            img = X[sample_index].astype(np.float32)
            img[:,:,0] -= 103.939
            img[:,:,1] -= 116.779
            img[:,:,2] -= 123.68
            #img = (img / 255. - .5).astype(np.float32)
            batch_X.append(img)
            batch_y.append(sa)
        yield np.array(batch_X), np.array(batch_y)



with open('./preprocessedData/data' + str(ROWS) + 'x' + str(COLS) + '.p', 'rb') as f:
    data = pickle.load(f)

X_all = data['feature']
y_all = data['label']

nb_epoch = 1
batch_size = 64

### Creating Validation Data
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=23)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='auto')
save_weights = ModelCheckpoint('./models/' + modelName + '/model.h5', monitor='val_loss', save_best_only=True)

model.fit_generator(fit_gen(batch_size, X_train, y_train),
        samples_per_epoch=len(X_train), nb_epoch=nb_epoch,
        validation_data=(np.array(X_test), np.array(y_test)), callbacks=[save_weights, early_stopping])


preds = model.predict(np.array(X_test), verbose=1)

print( "Test MSE: {}".format(mean_squared_error(np.array(y_test), preds)))
print( "Test RMSE: {}".format(np.sqrt(mean_squared_error(np.array(y_test), preds))))

with open('./models/' + modelName + '/model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

K.clear_session()