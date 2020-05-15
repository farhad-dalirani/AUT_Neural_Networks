###################################################################################
#   For Tensor board this http://desktop-cjgk7hj:6006 should be opend in chorme.  #
###################################################################################


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
from sklearn.model_selection import train_test_split
#################################################################################
#       Check GPU
#################################################################################
print(device_lib.list_local_devices())

#################################################################################
#       Open Tesor Board
#################################################################################
tensorBoardPath = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\logs'
def launchTensorBoard():
    import os
    print(os.system('cd'))
    #os.system('tensorboard --logdir=' + tensorBoardPath)
    os.system('tensorboard --logdir '+tensorBoardPath)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()

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
#       create and train autoencoder
#################################################################################
# create auto encoder
#autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 4500, 1875], decoder_layers=[4500, 96*96])
#autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 4500, 100], decoder_layers=[4500, 96*96])
#autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 4500, 20], decoder_layers=[4500, 96*96])
#autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 3000, 2900], decoder_layers=[3000, 96*96])
#autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 700, 500], decoder_layers=[700, 96*96])
#autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 900, 700, 500], decoder_layers=[700, 96*96])
autoencoder, encoder = create_auto_encoder(encoder_layers=[96*96, 1300, 1000, 900, 700, 500], decoder_layers=[700, 96*96])
#################################################################################
#       Train
#################################################################################
#autoencoder.fit(x_train,
#                x_train,
#                epochs=50,
#                batch_size=16,
#                shuffle=True,
#                validation_data=(x_valid, x_valid),
#                callbacks=[TensorBoard(log_dir=tensorBoardPath)])
autoencoder.fit(x_train,
                x_train,
                epochs=50,
                batch_size=8,
                shuffle=True,
                validation_data=(x_valid, x_valid),
                callbacks=[TensorBoard(log_dir=tensorBoardPath)])
#autoencoder.fit(x_train,
#                x_train,
#                epochs=15,
#                batch_size=1,
#                shuffle=True,
#                validation_data=(x_valid, x_valid),
#                callbacks=[TensorBoard(log_dir=tensorBoardPath)])

# evaluate the model
scores = autoencoder.evaluate(x_train, x_train, verbose=0)
print("Train Set Evaluation, {}: {}".format(autoencoder.metrics_names[0], scores[0]))
print("Train Set Evaluation, {}: {}".format(autoencoder.metrics_names[1], scores[1]))

scores = autoencoder.evaluate(x_valid, x_valid, verbose=0)
print("Validation Set Evaluation, {}: {}".format(autoencoder.metrics_names[0], scores[0]))
print("Validation Set Evaluation, {}: {}".format(autoencoder.metrics_names[1], scores[1]))

scores = autoencoder.evaluate(x_test, x_test, verbose=0)
print("Test Set Evaluation, {}: {}".format(autoencoder.metrics_names[0], scores[0]))
print("Test Set Evaluation, {}: {}".format(autoencoder.metrics_names[1], scores[1]))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

decoded_imgs = autoencoder.predict(x_test)


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

# serialize model to JSON
autoencoder_json = autoencoder.to_json()
with open("model.json", "w") as json_file:
    json_file.write(autoencoder_json)
# serialize weights to HDF5
autoencoder.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_autoencoder = model_from_json(loaded_model_json)
# load weights into new model
loaded_autoencoder.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['mse'])

# evaluate the model
scores = loaded_autoencoder.evaluate(x_train, x_train, verbose=0)
print("Train Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Train Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))

scores = loaded_autoencoder.evaluate(x_valid, x_valid, verbose=0)
print("Validation Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Validation Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))

scores = loaded_autoencoder.evaluate(x_test, x_test, verbose=0)
print("Test Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Test Set Evaluation, {}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))

t.join()
