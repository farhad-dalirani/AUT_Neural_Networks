import numpy as np
import warnings
from create_clean_dataset import create_clean_dataset
from sklearn.model_selection import train_test_split
from sklearn import decomposition


def read_dataset(train_ratio=0.80, valid_ratio=0.10, test_ratio=0.10, random_state=None, max_features=None):
    """
    Read 20 news category dataset. split it in train, validation and test
    :param train_ratio: portion of dataset that should be assigned to training
    :param valid_ratio: portion of dataset that should be assigned to validation
    :param test_ratio: portion of dataset that should be assigned to testing
    :param random_state: for controlling generating random nnumber and making function reproducible
    :return: trainX, validX, testX, trainY, validY, testY

            X format : ndarray [[observation1],[observation2],...,[observationN]]
            Y format : ndarray [[label observation1],[label observation2],...,[label observationN]]
    """

    # suppress future warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    trainX = None
    trainY = None
    validX = None
    validY = None
    testX = None
    testY = None

    # use seed for controlling generating random numbers
    if random_state is not None:
        np.random.seed(random_state)

    # create a clean dataset from files
    X, y, label_class = create_clean_dataset(max_features=max_features)

    # replace by one
    #X = np.where(X > 0, 1, 0)

    if X.shape[1] > 800:
        # pca
        pca = decomposition.PCA(n_components=800)
        pca.fit(X)
        X = pca.transform(X)

    # eliminate zero rows
    y = y[~(X==0).all(1)]
    X = X[~(X==0).all(1)]

    print('zero rows: ')
    zero_rows = np.sum(X, axis=1)
    zero_rows = np.where(zero_rows == 0, 1, 0)
    print(zero_rows.sum())
    print('===========================')

    # split date of number i
    # spiting data of number i to train, validation and test
    restX, temp_testX, restY, temp_testY = train_test_split(X, y,
                                                            train_size=train_ratio + valid_ratio,
                                                            random_state=random_state
                                                            )
    temp_trainX, temp_validX, temp_trainY, temp_validY = train_test_split(restX, restY,
                                                                          train_size=train_ratio / (
                                                                              train_ratio + valid_ratio),
                                                                          random_state=random_state
                                                                          )

    trainX = np.copy(temp_trainX)
    trainY = np.copy(temp_trainY)
    testX = np.copy(temp_testX)
    testY = np.copy(temp_testY)
    validX = np.copy(temp_validX)
    validY = np.copy(temp_validY)

    # shuffle train
    #permutation = np.random.permutation(trainX.shape[0])
    #trainX = trainX[permutation, :]
    #trainY = trainY[permutation, :]
    print('Categories Label: ')
    print(label_class)
    print('train: ',trainX.shape, trainY.shape)
    print('valid: ', validX.shape, validY.shape)
    print('test: ', testX.shape, testY.shape)



    return trainX, validX, testX, trainY, validY, testY

###################################
#       test functions for reading
#               dataset
###################################
if __name__ == '__main__':
    #trainX, validX, testX, trainY, validY, testY = read_digit(train_ratio=0.80, valid_ratio=0.10,
    #                                                          test_ratio=0.10, random_state=3)

    trainX, validX, testX, trainY, validY, testY = read_dataset(train_ratio=0.80, valid_ratio=0.10,
                                                              test_ratio=0.10, random_state=3, max_features=None)

    print('train: ', trainX.shape, trainY.shape)
    print('valid: ', validX.shape, validY.shape)
    print('test: ', testX.shape, testY.shape)

    for x in trainX[0]:
        print(x)