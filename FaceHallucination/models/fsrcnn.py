import numpy as np
from keras.optimizers import Adam, RMSprop
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv3D, MaxPooling3D, Activation, GlobalMaxPooling3D, MaxoutDense, SpatialDropout3D, GlobalAveragePooling3D, Conv2D, Conv2DTranspose, Conv3DTranspose
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, Callback
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from copy import deepcopy
from keras import backend as K
import os, os.path

def get_training_data():
  xdata = np.load('training/training_low_res.npy')
  ydata = np.load('training/training_high_res.npy')
  return xdata, ydata

def get_testing_data():
  xdata = np.load('testing/testing_low_res.npy')
  ydata = np.load('testing/testing_high_res.npy')

  return xdata, ydata

def split(datax, datay, frac):
  n_elems = int(frac * datax.shape[0])
  return datax[:n_elems], datay[:n_elems], datax[n_elems:], datay[n_elems:]


def PSNRLoss(y_true, y_pred):
    """
    PSNR is Peek Signal to Noise Ratio, which is similar to mean squared error.
    It can be calculated as
    PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    When providing an unscaled input, MAXp = 255. Therefore 20 * log10(255)== 48.1308036087.
    However, since we are scaling our input, MAXp = 1. Therefore 20 * log10(1) = 0.
    Thus we remove that component completely and only compute the remaining MSE component.
    """
    return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

CONST_DROPOUT = 0.4
CONST_BATCH_SIZE = 32
CONST_EPOCHS = 100
CONST_TEST_SPLIT = 0.1
CONST_VALID_SPLIT = 0.2

# Sensitive variables for tuning
D = 48 # Feature dimension of low resolution
S = 4 # Level of shrink
M = 3 # Number of mapping layers
UPSCALING_FACTOR = 4 # scaling factor for deconv
CHANNEL = 1

xtrain, ytrain = get_training_data()
NUM_PIXELS_X, NUM_SAMPLES = xtrain.shape
NUM_PIXELS_Y, NUM_SAMPLES = ytrain.shape

# Reshape
xtrain = xtrain.T.reshape(NUM_SAMPLES, 32, 24, CHANNEL)
ytrain = ytrain.T.reshape(NUM_SAMPLES, 128, 96, CHANNEL)

# 4 conv layer, 1 deconv
model = Sequential()

#Conv(5,d,1)
model.add(Conv2D(D, kernel_size=(5,5), input_shape=(32, 24, CHANNEL), padding='same'))
model.add(PReLU())

#Conv(1,s,d)
model.add(Conv2D(S, kernel_size=(1,1), padding='same'))
model.add(PReLU())

#Conv(3,s,s) x m
for i in range(M):
  model.add(Conv2D(S, kernel_size=(3,3), padding='same'))
  model.add(PReLU())

#Conv(1,d,s)
model.add(Conv2D(D, kernel_size=(1,1), padding='same'))
model.add(PReLU())

#DeConv(9,1,s)
model.add(Conv2DTranspose(1, kernel_size=(9,9), strides=(UPSCALING_FACTOR, UPSCALING_FACTOR), padding='same'))

print(model.summary())
optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer, metrics=[PSNRLoss])
history = model.fit(xtrain, ytrain, batch_size=CONST_BATCH_SIZE, epochs=CONST_EPOCHS, validation_split=CONST_VALID_SPLIT, shuffle=True,)

# xtest, ytest = get_testing_data()
# _, NUM_TESTING_SAMPLES = xtest.shape
# xtest = xtest.T.reshape(NUM_TESTING_SAMPLES, 32, 24, CHANNEL)
# ytest = ytest.T.reshape(NUM_TESTING_SAMPLES, 128, 96, CHANNEL)

model.save('fsrcnn.h5')

# Uncomment this to test on images
for path in os.listdir('result/low_res'):
  input_image = misc.imread(os.path.join('result/low_res', path), mode='L')
  input_image = input_image.reshape(1, 32, 24, 1)
  prediction = model.predict(input_image, batch_size=CONST_BATCH_SIZE)
  im = prediction[0, :, :, 0].reshape(128, 96)
  misc.imsave(os.path.join('result/hallucinated_cnn_100/', path), im)

# input_image = misc.imread('result/yang_low_res.png', mode='L')
# input_image = input_image.reshape(1, 32, 24, 1)
# prediction = model.predict(input_image, batch_size=CONST_BATCH_SIZE)
# im = prediction[0, :, :, 0].reshape(128, 96)
# plt.imshow(im, cmap='gray')
# plt.show()
# misc.imsave('result/yang_hallucinated.png', im)


