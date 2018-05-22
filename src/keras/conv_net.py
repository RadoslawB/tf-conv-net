from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import MaxPooling2D
from keras.models import Model


def conv_model(input_shape, classes):
    '''
    Building keras model via function API
    :param input_shape
    :param classes: number of possible output classes (OHO)
    :return: build model ready to compile and fit
    '''

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

    # Create model.
    model_ = Model(inputs=X_input, outputs=X, name='model')

    return model_