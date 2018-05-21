import pickle
import matplotlib.pyplot as plt


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def plot_imgs(imgs, columns, rows):
    fig = plt.figure(figsize=(columns, rows))

    print(imgs.shape)
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(imgs[i])
    plt.show()