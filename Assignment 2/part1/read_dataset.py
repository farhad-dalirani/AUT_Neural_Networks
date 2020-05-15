from scipy import misc
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def read_digit(train_ratio=0.80, valid_ratio=0.10, test_ratio=0.10, random_state=None):
    """
    Read Digit dataset.
    :param train_ratio: portion of dataset that should be assigned to training
    :param valid_ratio: portion of dataset that should be assigned to validation
    :param test_ratio: portion of dataset that should be assigned to testing
    :param random_state: for controlling generating random nnumber and making function reproducible
    :return: trainX, validX, testX, trainY, validY, testY

            X format : ndarray [[observation1],[observation2],...,[observationN]]
            Y format : ndarray [[label observation1],[label observation2],...,[label observationN]]
    """

    # suppress future warnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    max_label = -1

    trainX = None
    trainY = None
    validX = None
    validY = None
    testX = None
    testY = None

    # use seed for controlling generating random numbers
    if random_state is not None:
        np.random.seed(random_state)

    # read digits in different folders digit_0, digit_1, ....
    for num in range(0, 10):

        max_label = num

        print('Reading digit {}...'.format(num))
        imagesX = None
        imagesY = None

        # read images in folder digit_num
        for image_path in glob.glob(os.path.join('dataset','digit','digit_'+str(num),'*.png')):
            #print(image_path)

            # read and scale image
            image = misc.imread(image_path)/255

            # flat image
            image = image.reshape(1, -1)

            if imagesX is None:
                imagesX = np.copy(image)
                imagesY = np.copy(np.array([[num]])) # labbel
            else:
                imagesX = np.vstack((imagesX, image))
                imagesY = np.vstack((imagesY, np.array([[num]]))) # label

        # split date of number i
        # spiting data of number i to train, validation and test
        restX, temp_testX, restY, temp_testY = train_test_split(imagesX, imagesY,
                                                              train_size=train_ratio + valid_ratio,
                                                              random_state=random_state
                                                              )
        temp_trainX, temp_validX, temp_trainY, temp_validY = train_test_split(restX, restY,
                                                                  train_size=train_ratio / (train_ratio + valid_ratio),
                                                                  random_state=random_state
                                                                  )

        if trainX is None:
            trainX = np.copy(temp_trainX)
            trainY = np.copy(temp_trainY)
            testX = np.copy(temp_testX)
            testY = np.copy(temp_testY)
            validX = np.copy(temp_validX)
            validY = np.copy(temp_validY)
        else:
            trainX = np.vstack((trainX, temp_trainX))
            trainY = np.vstack((trainY, temp_trainY))
            testX = np.vstack((testX, temp_testX))
            testY = np.vstack((testY, temp_testY))
            validX = np.vstack((validX, temp_validX))
            validY = np.vstack((validY, temp_validY))

    # shuffle train
    permutation = np.random.permutation(trainX.shape[0])
    trainX = trainX[permutation, :]
    trainY = trainY[permutation, :]

    # transform labels to one hot
    unique_labels = np.arange(0, max_label+1).reshape(max_label+1, 1)
    transformer = OneHotEncoder()
    transformer.fit(unique_labels)
    trainY = transformer.transform(trainY).toarray()
    validY = transformer.transform(validY).toarray()
    testY = transformer.transform(testY).toarray()

    print('train: ',trainX.shape, trainY.shape)
    print('valid: ', validX.shape, validY.shape)
    print('test: ', testX.shape, testY.shape)

    return trainX, validX, testX, trainY, validY, testY


def read_digit_letter(train_ratio=0.80, valid_ratio=0.10, test_ratio=0.10, random_state=None):
    """
    Read  digits and letter dataset.
    :param train_ratio: portion of dataset that should be assigned to training
    :param valid_ratio: portion of dataset that should be assigned to validation
    :param test_ratio: portion of dataset that should be assigned to testing
    :param random_state: for controlling generating random nnumber and making function reproducible
    :return: trainX, validX, testX, trainY, validY, testY

            X format : ndarray [[observation1],[observation2],...,[observationN]]
            Y format : ndarray [[label observation1],[label observation2],...,[label observationN]]
    """

    # suppress future warnings
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    max_label = -1

    trainX = None
    trainY = None
    validX = None
    validY = None
    testX = None
    testY = None

    # use seed for controlling generating random numbers
    if random_state is not None:
        np.random.seed(random_state)

    max_label = None

    # read digits in different folders digit_0, digit_1, ....
    for num in range(0, 10):


        print('Reading digit {}...'.format(num))
        imagesX = None
        imagesY = None

        # read images in folder digit_num
        for image_path in glob.glob(os.path.join('dataset','digit','digit_'+str(num),'*.png')):
            #print(image_path)

            # read and scale image
            image = misc.imread(image_path)/255

            # flat image
            image = image.reshape(1, -1)

            if imagesX is None:
                imagesX = np.copy(image)
                imagesY = np.copy(np.array([[num]])) # labbel
            else:
                imagesX = np.vstack((imagesX, image))
                imagesY = np.vstack((imagesY, np.array([[num]]))) # label

        # split date of number i
        # spiting data of number i to train, validation and test
        restX, temp_testX, restY, temp_testY = train_test_split(imagesX, imagesY,
                                                                train_size=train_ratio + valid_ratio,
                                                                random_state=random_state
                                                                )
        temp_trainX, temp_validX, temp_trainY, temp_validY = train_test_split(restX, restY,
                                                                              train_size=train_ratio / (train_ratio + valid_ratio),
                                                                              random_state=random_state
                                                                              )

        if trainX is None:
            trainX = np.copy(temp_trainX)
            trainY = np.copy(temp_trainY)
            testX = np.copy(temp_testX)
            testY = np.copy(temp_testY)
            validX = np.copy(temp_validX)
            validY = np.copy(temp_validY)
        else:
            trainX = np.vstack((trainX, temp_trainX))
            trainY = np.vstack((trainY, temp_trainY))
            testX = np.vstack((testX, temp_testX))
            testY = np.vstack((testY, temp_testY))
            validX = np.vstack((validX, temp_validX))
            validY = np.vstack((validY, temp_validY))


    # read letter in different folders
    for num in range(1, 37):

        max_label = num + 9

        print('Reading letter {}...'.format(num))
        imagesX = None
        imagesY = None

        # read images in folder digit_num
        for image_path in glob.glob(os.path.join('dataset', 'character', 'character_' + str(num)+'_*', '*.png')):
            # print(image_path)

            # read and scale image
            image = misc.imread(image_path) / 255

            # flat image
            image = image.reshape(1, -1)

            if imagesX is None:
                imagesX = np.copy(image)
                imagesY = np.copy(np.array([[num+9]]))  # labbel
            else:
                imagesX = np.vstack((imagesX, image))
                imagesY = np.vstack((imagesY, np.array([[num+9]])))  # label

        # split date of number i
        # spiting data of number i to train, validation and test
        restX, temp_testX, restY, temp_testY = train_test_split(imagesX, imagesY,
                                                                train_size=train_ratio + valid_ratio,
                                                                random_state=random_state
                                                                )
        temp_trainX, temp_validX, temp_trainY, temp_validY = train_test_split(restX, restY,
                                                                              train_size=train_ratio / (
                                                                              train_ratio + valid_ratio),
                                                                              random_state=random_state
                                                                              )

        if trainX is None:
            trainX = np.copy(temp_trainX)
            trainY = np.copy(temp_trainY)
            testX = np.copy(temp_testX)
            testY = np.copy(temp_testY)
            validX = np.copy(temp_validX)
            validY = np.copy(temp_validY)
        else:
            trainX = np.vstack((trainX, temp_trainX))
            trainY = np.vstack((trainY, temp_trainY))
            testX = np.vstack((testX, temp_testX))
            testY = np.vstack((testY, temp_testY))
            validX = np.vstack((validX, temp_validX))
            validY = np.vstack((validY, temp_validY))

    # shuffle train
    permutation = np.random.permutation(trainX.shape[0])
    trainX = trainX[permutation, :]
    trainY = trainY[permutation, :]

    # transform labels to one hot
    unique_labels = np.arange(0, max_label+1).reshape(max_label+1, 1)
    transformer = OneHotEncoder()
    transformer.fit(unique_labels)
    trainY = transformer.transform(trainY).toarray()
    validY = transformer.transform(validY).toarray()
    testY = transformer.transform(testY).toarray()

    print('train: ',trainX.shape, trainY.shape)
    print('valid: ', validX.shape, validY.shape)
    print('test: ', testX.shape, testY.shape)

    """
    plt.figure()
    plt.imshow(trainX[300].reshape(32,32), cmap='gray')
    plt.title(trainY[300])
    plt.figure()
    plt.imshow(trainX[600].reshape(32,32), cmap='gray')
    plt.title(trainY[600])
    plt.figure()
    plt.imshow(trainX[900].reshape(32,32), cmap='gray')
    plt.title(trainY[900])
    plt.figure()
    plt.imshow(testX[600].reshape(32, 32), cmap='gray')
    plt.title(testY[600])
    plt.figure()
    plt.imshow(testX[900].reshape(32, 32), cmap='gray')
    plt.title(testY[900])
    plt.figure()
    plt.imshow(validX[600].reshape(32, 32), cmap='gray')
    plt.title(validY[600])
    plt.figure()
    plt.imshow(validX[900].reshape(32, 32), cmap='gray')
    plt.title(validY[900])
    plt.figure()
    plt.imshow(trainX[15000].reshape(32, 32), cmap='gray')
    plt.title(trainY[15000])
    plt.figure()
    plt.imshow(trainX[30000].reshape(32, 32), cmap='gray')
    plt.title(trainY[30000])
    plt.figure()
    plt.imshow(trainX[40000].reshape(32, 32), cmap='gray')
    plt.title(trainY[40000])
    plt.figure()
    plt.imshow(testX[2000].reshape(32, 32), cmap='gray')
    plt.title(testY[2000])
    plt.figure()
    plt.imshow(testX[4000].reshape(32, 32), cmap='gray')
    plt.title(testY[4000])
    plt.figure()
    plt.imshow(validX[3000].reshape(32, 32), cmap='gray')
    plt.title(validY[3000])
    plt.figure()
    plt.imshow(validX[5000].reshape(32, 32), cmap='gray')
    plt.title(validY[5000])
    plt.show()
    print(max_label)"""

    return trainX, validX, testX, trainY, validY, testY

###################################
#       test functions for reading
#               dataset
###################################
if __name__ == '__main__':
    #trainX, validX, testX, trainY, validY, testY = read_digit(train_ratio=0.80, valid_ratio=0.10,
    #                                                          test_ratio=0.10, random_state=3)

    trainX, validX, testX, trainY, validY, testY = read_digit_letter(train_ratio=0.80, valid_ratio=0.10,
                                                              test_ratio=0.10, random_state=3)

    print('train: ', trainX.shape, trainY.shape)
    print('valid: ', validX.shape, validY.shape)
    print('test: ', testX.shape, testY.shape)