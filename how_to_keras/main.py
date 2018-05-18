import matplotlib.pyplot as plt
import numpy as np
# from keras import layers
# import pickle
# import gzip
# from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
# from keras.models import Model
# from keras import optimizers
# from keras.preprocessing import image
# from keras.utils import layer_utils
# from keras.utils.data_utils import get_file
# from keras.applications.imagenet_utils import preprocess_input
# from IPython.display import SVG
# from keras.utils.vis_utils import model_to_dot
# from keras.utils import plot_model
import keras.backend as K
from utils import unpickle

K.set_image_data_format('channels_last')

from utils import unpickle
from model import model

data = unpickle('../cifar-10-batches/data_batch_1')

X_train_orig = data[b'data']
m = X_train_orig.shape[0]
# todo doczytaj o order
X_train = X_train_orig.reshape((m, 32, 32, 3), order='F')
Y_train_orig = data[b'labels']

index = 4544
columns = 4
rows = 5
h, w = 32, 32

# todo move to utils
fig = plt.figure(figsize=(8, 8))
for i in range(1, columns * rows + 1):
    fig.add_subplot(rows, columns, i)
    plt.imshow(X_train[i + index])
plt.show()

#
#
# def Model(input_shape):
#     # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
#     X_input = Input(input_shape)
#
#     # Zero-Padding: pads the border of X_input with zeroes
#     X = ZeroPadding2D((3, 3))(X_input)
#
#     # CONV -> BN -> RELU Block applied to X
#     X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
#     X = BatchNormalization(axis=3, name='bn0')(X)
#     X = Activation('relu')(X)
#
#     # MAXPOOL
#     X = MaxPooling2D((2, 2), name='max_pool')(X)
#
#     # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
#     X = Flatten()(X)
#     X = Dense(1, activation='sigmoid', name='fc')(X)
#
#     # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
#     model = Model(inputs=X_input, outputs=X, name='model')
#
#     return model
#
#
# model = Model(X_train.shape[1:])
# model.compile(optimizer=optimizers.Adam(), loss='mean_squared_error', metrics=['accuracy'])
