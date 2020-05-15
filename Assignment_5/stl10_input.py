####################################################
#           Standard code for work with dataset
#           which is recommended by stl-10 site.
#           with a little changes
####################################################

from __future__ import print_function

import sys
import os, sys, tarfile, errno
import numpy as np
import matplotlib.pyplot as plt

if sys.version_info >= (3, 0, 0):
    import urllib.request as urllib  # ugly but works
else:
    import urllib

print(sys.version_info)

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

# path to the directory with the data
DATA_DIR = './data'

# url of the binary data
DATA_URL = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'

# path to the binary train file with image data
DATA_PATH = './data/stl10_binary/train_X.bin'

# path to the binary train file with labels
LABEL_PATH = './data/stl10_binary/train_y.bin'


def read_labels(path_to_labels):
    """
    :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
    :return: an array containing the labels
    """
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def read_single_image(image_file):
    """
    CAREFUL! - this method uses a file as input instead of the path - so the
    position of the reader will be remembered outside of context of this method.
    :param image_file: the open file containing the images
    :return: a single image
    """
    # read a single image, count determines the number of uint8's to read
    image = np.fromfile(image_file, dtype=np.uint8, count=SIZE)
    # force into image matrix
    image = np.reshape(image, (3, 96, 96))
    # transpose to standard format
    # You might want to comment this line or reverse the shuffle
    # if you will use a learning algorithm like CNN, since they like
    # their channels separated.
    image = np.transpose(image, (2, 1, 0))
    return image


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def save_image(image, name):
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=True)
    plt.axis('off')

    plt.imshow(image)
    plt.savefig(name, bbox_inches='tight', dpi=96)


def download_and_extract():
    """
    Download and extract the STL-10 dataset
    :return: None
    """
    dest_directory = DATA_DIR
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                                                          float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.urlretrieve(DATA_URL, filepath, reporthook=_progress)
        print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)


def save_images(images, labels, path):
    print("Saving images to disk")
    i = 0
    for image in images:
        label = labels[i]
        directory = path
        try:
            os.mkdir(directory)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        print(filename)
        save_image(image, filename)
        i = i + 1


def read_names_of_labels(path):
    # read name of each numerical label
    file = open(path, 'r')
    labels_name = file.read()
    labels_name = labels_name.split()
    return labels_name


def images_rgb_to_gray(images):
    """
    convert RGB images to gray level images
    :param images:
    :return:
    """
    import numpy as np
    from PIL import Image
    # covert RGB images to gray level
    images_gray = np.zeros(shape=(images.shape[0], 96, 96), dtype=np.uint8)
    for i in range(0, images.shape[0]):
        images_gray[i, :] = np.dot(images[i], [0.299, 0.587, 0.114])
    return images_gray


def flatten_gray_images(images_gray):
    """
    Flatten gray images to 1-d
    :param images_gray: gray scale images
    :return:
    """
    # covert RGB images to gray level
    images_flat = np.zeros(shape=(images_gray.shape[0], 96*96), dtype=np.uint8)
    for i in range(0, images_gray.shape[0]):
        images_flat[i, :] = images_gray[i].reshape(1, 96*96)
    return images_flat


def images_gray_falt_version(images):
    """
    Convert RGB images to gray scale then faltten images
    :param images:
    :return:
    """
    # convert RGB images to gray level
    images = images_rgb_to_gray(images=images)
    # flat images
    images = flatten_gray_images(images_gray=images)
    # retrunt flatten gray-level images
    return images


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    # dataset path
    path_dataset = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary'
    path_train_x = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\train_X.bin'
    path_train_y = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\train_y.bin'
    path_test_x = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\test_X.bin'
    path_test_y = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\test_y.bin'
    path_unlabeled = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\unlabeled_X.bin'
    path_labels_name = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\class_names.txt'
    path_save_directory = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\new\\new'
    # download data if needed
    #download_and_extract()

    # test to check if the image is read correctly
    with open(path_train_x) as f:
        image = read_single_image(f)
        plot_image(image)
        print(type(image[0][0][0]))

    # Read images
    images = read_all_images(path_train_x)
    # convert images to gray flatten
    images = images_gray_falt_version(images=images)
    print(images.shape)

    # Read labels
    labels = read_labels(path_train_y)
    labels = labels-1
    print(labels.shape)

    # Read name of labels
    labels_name = read_names_of_labels(path=path_labels_name)
    print('labels name: {}'.format(labels_name))
    print(type(labels_name))

    # print some of images
    for i in range(0, 6):
        plt.figure()
        plt.imshow(images[i].reshape(96, 96), cmap='gray')
        plt.title(labels[i])
    plt.show()

    # save images to disk
    # save_images(images, labels, path_save_directory)