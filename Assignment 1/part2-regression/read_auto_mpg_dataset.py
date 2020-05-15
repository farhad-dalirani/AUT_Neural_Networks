def read_auto_mpg_dataset(rand_state=0, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15, order='1'):
    """
    This function reads auto-mpg dataset. split data to training, validation and test. it shuffles data.
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

    #print(train_ratio, valid_ratio, test_ratio)
    if train_ratio+test_ratio+valid_ratio != 1.0:
        raise ValueError('sum of Train ratio, Validation ratio and test ratio must be 1.0')

    # read auto-mpg dataset
    auto_mpg = pd.read_csv('auto-mpg.data', header=None, delim_whitespace=True, na_values='?')
    #print(auto_mpg.mean())
    #print(auto_mpg)
    #print(auto_mpg.shape)

    # fill missing data of each feature with its mean
    auto_mpg.fillna(value=auto_mpg.mean(), inplace=True)
    #print(auto_mpg.isnull().sum())

    # drop name column
    auto_mpg = auto_mpg.drop([8], axis=1)
    #print(auto_mpg)

    if order == '2':
        # generate terms of order 2 of continuous features
        k = 0
        auto_mpg_degree_2 = pd.DataFrame()
        for i in range(0, auto_mpg.shape[1]):
            auto_mpg_degree_2[k] = auto_mpg[i]
            k = k + 1

        #print('>',k)
        #continuous_features_index = [2, 3, 4, 5]
        for i in range(2, 6):
            for j in range(i, 6):
                #print('>> ',i, j)
                auto_mpg_degree_2[k] = auto_mpg[i] * auto_mpg[j]
                k = k + 1
        #print(auto_mpg_degree_2.head())

        # separate features and labels and convert to numpy array
        auto_mpg_features = auto_mpg_degree_2.iloc[:, 1:18].values
        auto_mpg_labels = auto_mpg_degree_2.iloc[:, 0:1].values
        #print(auto_mpg_degree_2.iloc[:, 1:18].head())
        #print(auto_mpg_degree_2.iloc[:, 1:18].head())
        #print(auto_mpg_degree_2.iloc[:, 0:1].head())

    elif order == '1':
        # separate features and labels and convert to numpy array
        auto_mpg_features = auto_mpg.iloc[:, 1:8].values
        auto_mpg_labels = auto_mpg.iloc[:, 0:1].values

    else:
        raise ValueError('order should be 1 or 2.')

    # spiting dataset to train, validation and test
    restX, testX, restY, testY = train_test_split(auto_mpg_features, auto_mpg_labels,
                                                    train_size=train_ratio+valid_ratio,
                                                    random_state=rand_state
                                                    )
    trainX, validX, trainY, validY = train_test_split(restX, restY,
                                                  train_size=train_ratio/(train_ratio + valid_ratio),
                                                  random_state=rand_state
                                                  )

    return trainX, validX, testX, trainY, validY, testY


def read_auto_mpg_dataset_continuous(rand_state=0, train_ratio=0.70, valid_ratio=0.15, test_ratio=0.15, order='1'):
    """
    This function reads auto-mpg dataset. split data to training, validation and test. it shuffles data.
    It omits categorical data
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

    #print(train_ratio, valid_ratio, test_ratio)
    if train_ratio+test_ratio+valid_ratio != 1.0:
        raise ValueError('sum of Train ratio, Validation ratio and test ratio must be 1.0')

    # read auto-mpg dataset
    auto_mpg = pd.read_csv('auto-mpg.data', header=None, delim_whitespace=True, na_values='?')
    #print(auto_mpg.mean())
    #print(auto_mpg)
    #print(auto_mpg.shape)

    # fill missing data of each feature with its mean
    auto_mpg.fillna(value=auto_mpg.mean(), inplace=True)
    #print(auto_mpg.isnull().sum())

    # drop name column
    auto_mpg = auto_mpg.drop([8], axis=1)
    #print(auto_mpg)

    # drop categorical data
    auto_mpg = auto_mpg.drop([1, 6, 7], axis=1)
    auto_mpg.columns = [i for i in range(0, auto_mpg.shape[1])]
    #print(auto_mpg.head())

    if order == '2':
        # generate terms of order 2 of continuous features
        k = 0
        auto_mpg_degree_2 = pd.DataFrame()
        for i in range(0, auto_mpg.shape[1]):
            auto_mpg_degree_2[k] = auto_mpg[i]
            k = k + 1

        print('> ', k)
        #continuous_features_index = [2, 3, 4, 5]
        for i in range(1, 5):
            for j in range(i, 5):
                print(k, '>> ', i, j)
                auto_mpg_degree_2[k] = auto_mpg[i] * auto_mpg[j]
                k = k + 1
        #print(auto_mpg_degree_2.head())

        # separate features and labels and convert to numpy array
        auto_mpg_features = auto_mpg_degree_2.iloc[:, 1:18].values
        auto_mpg_labels = auto_mpg_degree_2.iloc[:, 0:1].values
        #print(auto_mpg_degree_2.iloc[:, 1:18].head())
        #print(auto_mpg_degree_2.iloc[:, 0:1].head())

    elif order == '1':
        # separate features and labels and convert to numpy array
        auto_mpg_features = auto_mpg.iloc[:, 1:5].values
        #print(auto_mpg.iloc[:, 1:5])
        auto_mpg_labels = auto_mpg.iloc[:, 0:1].values
        #print(auto_mpg.iloc[:, 1:5].head())
        #print(auto_mpg.iloc[:, 0:1].head())
    else:
        raise ValueError('order should be 1 or 2.')

    # spiting dataset to train, validation and test
    restX, testX, restY, testY = train_test_split(auto_mpg_features, auto_mpg_labels,
                                                    train_size=train_ratio+valid_ratio,
                                                    random_state=rand_state
                                                    )
    trainX, validX, trainY, validY = train_test_split(restX, restY,
                                                  train_size=train_ratio/(train_ratio + valid_ratio),
                                                  random_state=rand_state
                                                  )

    return trainX, validX, testX, trainY, validY, testY


if __name__ == '__main__':
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

    trainX, validX, testX, trainY, validY, testY = read_auto_mpg_dataset(rand_state=0, train_ratio=0.70, valid_ratio=0.15,
                                                                         test_ratio=0.15, order='2')

    #trainX, validX, testX, trainY, validY, testY = read_auto_mpg_dataset_continuous(rand_state=0, train_ratio=0.70,
    #                                                                     valid_ratio=0.15,
    #                                                                     test_ratio=0.15, order='2')

    #print(trainX, validX, testX, trainY, validY, testY)
