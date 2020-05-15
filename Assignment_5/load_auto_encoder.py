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
path = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder10\\'

# load json and create model
json_file = open(path+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_autoencoder = model_from_json(loaded_model_json)
# load weights into new model
loaded_autoencoder.load_weights(path+"model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])

# evaluate the model
scores = loaded_autoencoder.evaluate(x_train, x_train, verbose=0)
print("Train Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Train Set Evaluation, {}: {}\n".format(loaded_autoencoder.metrics_names[1], scores[1]))

scores = loaded_autoencoder.evaluate(x_valid, x_valid, verbose=0)
print("Validation Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Validation Set Evaluation, {}: {}\n".format(loaded_autoencoder.metrics_names[1], scores[1]))

scores = loaded_autoencoder.evaluate(x_test, x_test, verbose=0)
print("Test Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Test Set Evaluation, {}: {}\n".format(loaded_autoencoder.metrics_names[1], scores[1]))

decoded_imgs = loaded_autoencoder.predict(x_test)
n = 10  # how many image will be displayed
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(96, 96))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(96, 96))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

print("Config: \n{}".format(loaded_autoencoder.get_config()))
weights = loaded_autoencoder.get_weights()
print(len(weights))

for layer_i in range(len(weights)):
    if layer_i % 2 == 1:
        continue
    print(weights[layer_i].shape)
    plt.figure()

    plt.imshow(weights[layer_i]/(np.max(np.max(weights[layer_i]))))
    plt.gray()
    plt.show()

neurons_value = np.ones((96*96, len(weights)//2))
k = 0
max_dim = -1
for layer_i in range(len(weights)):
    if len(weights[layer_i].shape) == 2:
        continue
    print(weights[layer_i].shape)
    if weights[layer_i].shape[0] > max_dim and weights[layer_i].shape[0] != 96*96:
        max_dim = weights[layer_i].shape[0]

    neurons_value[0:weights[layer_i].shape[0], k] = weights[layer_i]
    k = k+1

neurons_value = neurons_value[0:max_dim, :]
neurons_value = neurons_value[:, 0:k-1]
from skimage.transform import resize
neurons_value = resize(neurons_value, (1000,  300))

print(neurons_value.shape)
plt.figure()
plt.imshow(neurons_value)
plt.gray()
plt.show()
