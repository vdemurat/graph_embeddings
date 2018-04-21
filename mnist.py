import os
import struct
import numpy as np
import matplotlib.pyplot as plt

"""
read() modified from https://gist.github.com/akesling/5358964
"""

def read(path = "."):
    """
    Returns an iterator of 2-tuples (label, image)
    """
    fname_img = os.path.join(path, 'train-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)

        
        
def get_mnist_data(path, nb_img_per_digit=100):
    images, labels = [], []
    for label, img in read(path):
        images.append(img.reshape(1,28*28).reshape(-1))
        labels.append(label)
    labels = np.array(labels)
    
    data, digits = [], np.array([])
    for i in range(10):
        to_get = np.where(labels==i)[0][:nb_img_per_digit]
        #digits = np.hstack((digits, labels[to_get]))
        for j in to_get:
            data.append(images[j])        
    del images, labels
    return data
    
        

# useless for us (SUPPR ?)
def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    import matplotlib as mpl
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.show()