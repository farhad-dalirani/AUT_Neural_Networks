from tensorflow.python.client import device_lib
from keras.callbacks import TensorBoard
import threading
from keras.layers import Input, Dense
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from auto_encoders import create_auto_encoder
from keras.models import Sequential
from keras.models import model_from_json
from stl10_input import read_all_images, images_gray_falt_version, read_labels, read_names_of_labels
import matplotlib.pyplot as plt

#################################################################################
#       Read Dataset
#################################################################################
# dataset paths
path_dataset = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary'
path_train_x = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\train_X.bin'
path_train_y = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\train_y.bin'
path_test_x = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\test_X.bin'
path_test_y = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\test_y.bin'
path_unlabeled = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\unlabeled_X.bin'
path_labels_name = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\class_names.txt'
path_save_directory = 'D:\\ANN\\dataset\\stl10_binary\\stl10_binary\\new\\new'
from sklearn.model_selection import train_test_split

# Read images (train images)
x_train = read_all_images(path_train_x)
# convert images to gray flatten
x_train = images_gray_falt_version(images=x_train)
# convert type
x_train = x_train.astype('float32') / 255.
print(x_train.shape, type(x_train))

# Read images (test images)
x_test = read_all_images(path_test_x)
# convert images to gray flatten
x_test = images_gray_falt_version(images=x_test)
# convert type
x_test = x_test.astype('float32') / 255.
print(x_test.shape, type(x_test))

# Read labels (train)
y_train = read_labels(path_train_y)
print(y_train.shape)

# Read labels (train)
y_test = read_labels(path_test_y)
print(y_test.shape)

# train validation
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.80, random_state=64)
print(x_train.shape, y_train.shape, x_valid.shape, y_valid.shape)

# Read name of labels
labels_name = read_names_of_labels(path=path_labels_name)
print('labels name: {}'.format(labels_name))
print(type(labels_name))

"""
# print some of images(train)
for i in range(0, 6):
    plt.figure()
    plt.imshow(x_train[i].reshape(96, 96), cmap='gray')
    plt.title(y_train[i])
plt.show()
# print some of images(test)
for i in range(0, 6):
    plt.figure()
    plt.imshow(x_test[i].reshape(96, 96), cmap='gray')
    plt.title(y_test[i])
plt.show()
"""


#################################################################################
#       Load Auto encoder from file
#################################################################################
# model 11
#path_1 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder10\\'
#path_2 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder10\\'

# model 12
#path_1 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder1\\'
#path_2 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder1\\'

# model 13
path_1 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder3\\'
path_2 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder3\\'

# model 12
#path_1 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder1\\'
#path_2 = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder8\\'

# load json and create model
json_file_1 = open(path_1 + 'model.json', 'r')
json_file_2 = open(path_2 + 'model.json', 'r')
loaded_model_json_1 = json_file_1.read()
loaded_model_json_2 = json_file_2.read()
json_file_1.close()
json_file_2.close()
loaded_autoencoder_1 = model_from_json(loaded_model_json_1)
loaded_autoencoder_2 = model_from_json(loaded_model_json_2)

# load weights into new model
loaded_autoencoder_1.load_weights(path_1 + "model.h5")
loaded_autoencoder_2.load_weights(path_2 + "model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_autoencoder_1.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])
loaded_autoencoder_2.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])


# evaluate the model
decoded_x_train = loaded_autoencoder_1.predict(x_train)
decoded_x_train = loaded_autoencoder_2.predict(decoded_x_train)
mse_score = np.sqrt(np.sum(np.sum(((x_train-decoded_x_train)**2)))) / x_train.shape[0]
print('MSE SCORE OF TRAIN: {}'.format(mse_score))

# evaluate the model
decoded_x_valid = loaded_autoencoder_1.predict(x_valid)
decoded_x_valid = loaded_autoencoder_2.predict(decoded_x_valid)
mse_score = np.sqrt(np.sum(np.sum(((x_valid-decoded_x_valid)**2)))) / x_valid.shape[0]
print('MSE SCORE OF Validation: {}'.format(mse_score))

# evaluate the model
decoded_x_test = loaded_autoencoder_1.predict(x_test)
decoded_x_test = loaded_autoencoder_2.predict(decoded_x_test)
mse_score = np.sqrt(np.sum(np.sum(((x_test-decoded_x_test)**2)))) / x_test.shape[0]
print('MSE SCORE OF TEST: {}'.format(mse_score))

n = 10  # how many image will be displayed
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_train[i].reshape(96, 96))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_x_train[i].reshape(96, 96))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
