from keras import optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from src.utils import unpickle
import keras
import subprocess
from keras.callbacks import LambdaCallback
data = unpickle('../../cifar-10-batches/data_batch_1')
from src.utils import plot_imgs
X_train_orig = data[b'data']
Y_train_orig = data[b'labels']
m = X_train_orig.shape[0]

X_train = X_train_orig.reshape((m, 32, 32, 3), order='F') / 255
classes = 10
Y_train = to_categorical(Y_train_orig, classes)
print(Y_train.shape)


# plot_imgs(X_train, columns=4, rows=4)

def conv_model(input_shape, classes):
    '''
    Building keras model via function API
    :param input_shape
    :param classes: number of possible output classes (OHO)
    :return: build model ready to compile and fit
    '''
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = Conv2D(25, (3, 3), strides=(1, 1), padding='same', name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1024, activation='relu', name='fc0')(X)
    X = Dense(classes, activation='sigmoid', name='fc1')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model_ = Model(inputs=X_input, outputs=X, name='model')

    return model_


model = conv_model(X_train.shape[1:], classes)
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 3
batch_size = 512
model.save('model_epochs-' + str(epochs))

if __name__ == '__main__':
    print(model.summary())
    # todo https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/ - keras.callback lifecycle
    tf_board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=True, embeddings_metadata=True)

    logs_dir = './src/keras/logs'
    subprocess.run('tensorboard --logdir=' + logs_dir, shell=True)
    model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=batch_size, callbacks=[tf_board])
    print(model.summary())
