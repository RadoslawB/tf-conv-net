from keras import optimizers
from keras.utils import to_categorical
from src.utils import unpickle
import keras
import subprocess
from src.utils import plot_imgs
from src.keras.conv_net import conv_model
from src.keras.res_net import ResNet50
data = unpickle('../../cifar-10-batches/test_batch')

X_train_orig = data[b'data']
Y_train_orig = data[b'labels']
m = X_train_orig.shape[0]

X_train = X_train_orig.reshape((m, 32, 32, 3), order='F') / 255
classes = 10
Y_train = to_categorical(Y_train_orig, classes)
# plot_imgs(X_train, columns=4, rows=4)

# model = conv_model(X_train.shape[1:], classes)
model = ResNet50(X_train.shape[1:], classes)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


epochs = 3
batch_size = 512

if __name__ == '__main__':
    print(model.summary())
    # todo https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/ - keras.callback lifecycle
    tf_board = keras.callbacks.TensorBoard(log_dir='./logs', batch_size=batch_size, write_graph=True)

    logs_dir = './src/keras/logs'
    model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, callbacks=[tf_board])
    # subprocess.run('tensorboard --logdir=' + logs_dir, shell=True)

    model.save('model_epochs-' + str(epochs))
