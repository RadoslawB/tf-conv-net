from keras import optimizers
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model
from keras.utils import to_categorical
from src.utils import unpickle

data = unpickle('../cifar-10-batches/data_batch_1')

X_train_orig = data[b'data']
Y_train_orig = data[b'labels']
m = X_train_orig.shape[0]

X_train = X_train_orig.reshape((m, 32, 32, 3), order='F') / 255
classes = 10
Y_train = to_categorical(Y_train_orig, classes)
print(Y_train.shape)


# plot_imgs(X_train, columns=4, rows=4)


def conv_model(input_shape, classes):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (5, 5), strides=(1, 1), padding='same', name='conv0')(X_input)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Conv2D(25, (3, 3), strides=(1, 1), padding='same', name='conv1')(X_input)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1024, activation='relu', name='fc0')(X)
    X = Dense(classes, activation='sigmoid', name='fc1')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model_ = Model(inputs=X_input, outputs=X, name='model')

    return model_


model = conv_model(X_train.shape[1:], classes)
model.compile(optimizer=optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 1

model.save('model_epochs-' + str(epochs))

if __name__ == '__main__':
    print(model.summary())
    model.fit(x=X_train, y=Y_train, epochs=epochs, batch_size=512)
    print(model.summary())