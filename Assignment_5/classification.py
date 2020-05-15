import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
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
import keras
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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
#       Open Tesor Board
#################################################################################
tensorBoardPath = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\classification_logs'
def launchTensorBoard():
    import os
    print(os.system('cd'))
    #os.system('tensorboard --logdir=' + tensorBoardPath)
    os.system('tensorboard --logdir '+tensorBoardPath)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()


#################################################################################
#       Load Best autoencoder Autoencoder from file and transform images
#       by autoencoder
#################################################################################
# path of best auto encoder
path = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\Assignment_5\\autoencoder6\\'

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
print("Train Evaluation {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Train Evaluation {}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))

scores = loaded_autoencoder.evaluate(x_test, x_test, verbose=0)
print("Test Evaluation {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Test Evaluation {}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))

scores = loaded_autoencoder.evaluate(x_valid, x_valid, verbose=0)
print("Validation Evaluation {}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))
print("Validation Evaluation {}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))

# use auto encoder to compress train and test data
x_train = loaded_autoencoder.predict(x_train)
x_valid = loaded_autoencoder.predict(x_valid)
x_test = loaded_autoencoder.predict(x_test)

#################################################################################
#       Create MLP
#################################################################################
import numpy as np

# convert labels to one hot
y_train = y_train-1
y_train_not_one_hot = y_train
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_valid = y_valid-1
y_valid_not_one_hot = y_valid
y_valid = keras.utils.to_categorical(y_valid, num_classes=10)
y_test = y_test-1
y_test_not_one_hot = y_test
y_test = keras.utils.to_categorical(y_test, num_classes=10)


number_of_hidden_units = 2000
mlp = Sequential()
# dense fully connected layer
mlp.add(Dense(number_of_hidden_units, activation='relu', input_dim=96*96))
mlp.add(Dropout(0.5))
mlp.add(Dense(number_of_hidden_units, activation='relu'))
mlp.add(Dropout(0.5))
mlp.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
mlp.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

mlp.fit(x_train, y_train,
          epochs=410,
          batch_size=1000,
          validation_data=(x_valid, y_valid),
          callbacks=[TensorBoard(log_dir=tensorBoardPath)])


# evaluate the model
#score = mlp.evaluate(x_train, y_train, batch_size=1)
#print("Train Evaluation {}: {}".format(mlp.metrics_names[0], scores[0]))
#print("Train Evaluation {}: {}".format(mlp.metrics_names[1], scores[1]))

#score = mlp.evaluate(x_valid, y_valid, batch_size=1)
#print("Validatioin Evaluation {}: {}".format(mlp.metrics_names[0], scores[0]))
#print("Validatioin Evaluation {}: {}".format(mlp.metrics_names[1], scores[1]))

#score = mlp.evaluate(x_test, y_test, batch_size=1)
#print("Test Evaluation {}: {}".format(mlp.metrics_names[0], scores[0]))
#print("Test Evaluation {}: {}".format(mlp.metrics_names[1], scores[1]))


#################################################################################
#       Confusion matrix by sklearn
#################################################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.OrRd):
    """
         ********************************
         ***** FUNCTION BY Sklearn ******
         ********************************
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#y_test_pre = mlp.predict(x_test)
y_test_pre = mlp.predict_classes(x_test)

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test_not_one_hot, y_test_pre)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=labels_name, normalize=True,
                      title='Normalized confusion matrix')




#################################################################################
#       SAVE MODEL IN FILE
#################################################################################
# save model in file
# serialize model to JSON
mlp_json = mlp.to_json()
with open("model_classification.json", "w") as json_file:
    json_file.write(mlp_json)
# serialize weights to HDF5
mlp.save_weights("model_classification.h5")
print("Saved model to disk")

# later...

# load json and create model
#json_file = open('model_classification.json', 'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_mlp = model_from_json(loaded_model_json)
# load weights into new model
#loaded_mlp.load_weights("model_classification.h5")
#print("Loaded model from disk")

# evaluate loaded model on test data
#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#loaded_mlp.compile(loss='categorical_crossentropy',
#              optimizer=sgd,
#              metrics=['accuracy'])

# evaluate the model
#score = loaded_mlp.evaluate(x_train, y_train, batch_size=1000)
#print("Train Evaluation {}: {}".format(loaded_mlp.metrics_names[0], scores[0]))
#print("Train Evaluation {}: {}".format(loaded_mlp.metrics_names[1], scores[1]))

#score = loaded_mlp.evaluate(x_valid, y_valid, batch_size=1000)
#print("Validatioin Evaluation {}: {}".format(loaded_mlp.metrics_names[0], scores[0]))
#print("Validatioin Evaluation {}: {}".format(loaded_mlp.metrics_names[1], scores[1]))

#score = loaded_mlp.evaluate(x_test, y_test, batch_size=1000)
#print("Test Evaluation {}: {}".format(loaded_mlp.metrics_names[0], scores[0]))
#print("Test Evaluation {}: {}".format(loaded_mlp.metrics_names[1], scores[1]))

plt.show()
t.join()
