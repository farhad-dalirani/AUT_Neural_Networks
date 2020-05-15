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

print(device_lib.list_local_devices())

tensorBoardPath = 'C:\\Users\\Home\\Dropbox\\codes\\ANN\\autoencoder_MINST\\logs'
def launchTensorBoard():
    import os
    print(os.system('cd'))
    #os.system('tensorboard --logdir=' + tensorBoardPath)
    os.system('tensorboard --logdir '+tensorBoardPath)
    return

t = threading.Thread(target=launchTensorBoard, args=([]))
t.start()


# Read Dataset
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

# create auto encoder
#autoencoder, encoder, decoder = create_auto_encoder(encoder_layers=[784, 32, 8], decoder_layers=[16, 784])
# create auto encoder
autoencoder, encoder = create_auto_encoder(encoder_layers=[784, 32, 8], decoder_layers=[30, 784])


autoencoder.fit(x_train,
                x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir=tensorBoardPath)])

# evaluate the model
scores = autoencoder.evaluate(x_test, x_test, verbose=0)
print("{}: {}".format(autoencoder.metrics_names[1], scores[1]))
print("{}: {}".format(autoencoder.metrics_names[0], scores[0]))

# encode and decode some digits
# note that we take them from the *test* set
encoded_imgs = encoder.predict(x_test)
#decoded_imgs = decoder.predict(encoded_imgs)

decoded_imgs = autoencoder.predict(x_test)

# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
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
scores = loaded_autoencoder.evaluate(x_test, x_test, verbose=0)
print("{}: {}".format(loaded_autoencoder.metrics_names[1], scores[1]))
print("{}: {}".format(loaded_autoencoder.metrics_names[0], scores[0]))

t.join()