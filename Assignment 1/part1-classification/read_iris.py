def read_iris_dataset(rand_state=0, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15, order='1'):
    """
    This function reads Iris dataset, replace string class name with
    0, 1, 2. split data to training, validation and test. it shuffles data.
    :param rand_state: seed for shuffling
    :param train_ratio: portion of data that should be separated for training.
    :param valid_ratio: portion of data that should be separated for validation.
    :param test_ratio: portion of data that should be separated for testing.
    :param order: if degree is '1' read data set and return it, if it's '2'
                    generate order 2 of features (XiXj) then return data set.
    :return: trainX, validX, testX, trainY, validY, testY
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    #print(train_ratio, valid_ratio, test_ratio)
    if train_ratio+test_ratio+valid_ratio != 1.0:
        raise ValueError('sum of Train ratio, Validation ratio and test ratio must be 1.0')

    # read iris dataset
    iris = pd.read_csv('iris.data', header=None)
    #print(iris.head())
    #print(iris.shape)

    if order == '2':
        # generate terms of order 2 of features
        k = 0
        iris_degree_2 = pd.DataFrame()
        for i in range(0, iris.shape[1] - 1):
            #print(i)
            iris_degree_2[k] = iris[i]
            k = k + 1

        for i in range(0, iris.shape[1] - 1):
            for j in range(i, iris.shape[1] - 1):
                #print(i, j)
                iris_degree_2[k] = iris[i] * iris[j]
                k = k + 1

        #print(iris.shape[1] - 1)
        iris_degree_2[k] = iris[iris.shape[1] - 1]
        iris = iris_degree_2
        #print(iris.head())

        # separate features and labels and convert to numpy array
        iris_features = iris.iloc[:, 0:14].values
        iris_labels = iris.iloc[:, 14:15].values
    elif order == '1':
        # separate features and labels and convert to numpy array
        iris_features = iris.iloc[:, 0:4].values
        iris_labels = iris.iloc[:, 4:5].values
    else:
        raise ValueError('order should be 1 or 2.')

    # spiting dataset to train, validation and test
    restX, testX, restY, testY = train_test_split(iris_features, iris_labels,
                                                    train_size=train_ratio+valid_ratio,
                                                    random_state=rand_state
                                                    )
    trainX, validX, trainY, validY = train_test_split(restX, restY,
                                                  train_size=train_ratio/(train_ratio + valid_ratio),
                                                  random_state=rand_state
                                                  )

    return trainX, validX, testX, trainY, validY, testY


def read_iris_dataset_FLNN(rand_state=0, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15):
    """
    This function reads Iris dataset, replace string class name with
    0, 1, 2. split data to training and test. also it
    shuffles data. and generate Functional link NN terms
    :param rand_state: seed for shuffling
    :param train_ratio: portion of data that should be separated for training.
    :param valid_ratio: portion of data that should be separated for validation.
    :param test_ratio: portion of data that should be separated for testing.
    :param order: if degree is '1' read data set and return it, if it's '2'
                    generate order 2 of features XiXj then return data set.
    :return: trainX, validX, testX, trainY, validY, testY
    """
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split

    # read iris dataset
    iris = pd.read_csv('iris.data', header=None)
    #print(iris.head())

    if train_ratio+test_ratio+valid_ratio != 1.0:
        raise ValueError('sum of Train ratio, Validation ratio and test ratio must be 1.0')

    # generate terms of order 2 of features
    k = 0
    iris_degree_2 = pd.DataFrame()
    for i in range(0, iris.shape[1] - 1):
        #print(i)
        iris_degree_2[k] = iris[i]
        k = k + 1

    #add xi xj  (i<j)
    for i in range(0, iris.shape[1] - 1):
        for j in range(i+1, iris.shape[1] - 1):
            #print(i, j)
            iris_degree_2[k] = iris[i] * iris[j]
            k = k + 1

    #print(iris.shape[1] - 1)
    iris_degree_2[k] = iris[iris.shape[1] - 1]
    iris = iris_degree_2
    #print(iris.head())

    # separate features and labels and convert to numpy array
    iris_features = iris.iloc[:, 0:10].values
    iris_labels = iris.iloc[:, 10:11].values

    # spiting dataset to train, validation and test
    restX, testX, restY, testY = train_test_split(iris_features, iris_labels,
                                                  train_size=train_ratio + valid_ratio,
                                                  random_state=rand_state
                                                  )
    trainX, validX, trainY, validY = train_test_split(restX, restY,
                                                      train_size=train_ratio/(train_ratio + valid_ratio),
                                                      random_state=rand_state
                                                      )

    return trainX, validX, testX, trainY, validY, testY





