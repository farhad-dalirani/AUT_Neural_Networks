if __name__ == '__main__':

    # Import packages
    import numpy as np
    from measures import purity_measure, rand_index, f_measure
    import time
    import copy
    import sys
    from read_dataset import read_dataset
    import json
    from SOM import SOM
    import threading

    # get inputs
    print('Do want to give parameters for training SOM via a file or you want to load a trained SOM that stored in file:')
    print('>    1. I want to give parameters via file')
    print('>    2. I want to give a trained SOM network via file')
    train_load = int(float(input('Enter 1 or 2: ')))

    if train_load != 1 and train_load != 2:
        raise ValueError('Input should be 1 or 2!')

    if train_load == 1:
        # read parameter via file
        path_config = str(input('\nEnter name of the config file that contains necessary parameters for training SOM.'+
                                '\n(It should be located in directory of code. Example: input.json) : '))
        # read json file
        try:
            data = json.load(open(path_config))
        except:
            print('Can\'t open input json file!')
            exit(2)

        ratio_of_train_set = float(data['ratio_of_train_set']) # (0.0, 1.0)
        ratio_of_valid_set = float(data['ratio_of_valid_set']) # (0.0, 1.0)
        ratio_of_test_set = float(data['ratio_of_test_set']) # (0.0, 1.0)
        initial_learning_rate = float(data['initial_learning_rate']) # positive number
        max_epochs = int(float(data['max_epochs'])) # integer number
        cut_cost = float(data['cut_cost']) # positive float number for stop training when cost less than it
        if cut_cost == -1:
            cut_cost = None
        random_state = int(float(data['random_state'])) # integer number

        som_shape = data['shape_som'] # a list
        type_distance = data['type_distance'] # a list of activation type in each layer like ['tanh',...,'tanh', 'sigmoid']
        type_of_neighbourhood = str(data['type_of_neighbourhood'])  # a string 'MSE' or 'cross_entropy'

    else:
        path_trained_som = str(input('\nEnter name of the json file that contains a trained SOM.' +
                                     '\n(It should be located in directory of code. Example: input.json) : '))

        ratio_of_train_set = float(input('What ratio of dataset should be used for training(0.0, 1.0): ')) # (0.0, 1.0)
        ratio_of_valid_set = float(input('What ratio of dataset should be used for validation(0.0, 1.0): ')) # (0.0, 1.0)
        ratio_of_test_set = float(input('What ratio of dataset should be used for testing(0.0, 1.0): ')) # (0.0, 1.0)
        random_state = int(float(input('Enter an integer for seeding random numbers: ')))

    # read dataset
    print('reading dataset ...\n')
    trainX, validX, testX, trainY, validY, testY = read_dataset(train_ratio=ratio_of_train_set, valid_ratio=ratio_of_valid_set,
                                                                  test_ratio=ratio_of_test_set, random_state=random_state)

    som = None
    if train_load == 1:
        #print(som_shape, type_distance, type_of_neighbourhood, max_epochs, initial_learning_rate, ratio_of_train_set, ratio_of_valid_set, ratio_of_test_set)
        # select some samples as random weights fo initialization
        initial_sample_indexes = np.random.permutation(trainX.shape[0])
        number_of_neurons = som_shape[0] * som_shape[1] * som_shape[2]

        # create & train SOM
        som = SOM(shape=som_shape, number_of_feature=trainX.shape[1],
                  distance_measure_str=type_distance, topology=type_of_neighbourhood,
                  init_learning_rate=initial_learning_rate,
                  max_epoch=max_epochs,
                  samples_for_init=trainX[initial_sample_indexes[0:number_of_neurons]])
        # train
        som.fit(trainX)

    else:
        # load som from file
        # create a dummy SOM then load orginal SOM from file
        som = SOM(shape=(2,2,2), number_of_feature=trainX.shape[1], samples_for_init=np.zeros((8, trainX.shape[1])))
        # load
        som.load_from_file(file_name=path_trained_som)

    y_train_pre = som.predict(X=trainX)
    y_valid_pre = som.predict(X=validX)
    y_test_pre = som.predict(X=testX)

    y_train_pre = y_train_pre.reshape(y_train_pre.shape[0], 1)
    y_valid_pre = y_valid_pre.reshape(y_valid_pre.shape[0], 1)
    y_test_pre = y_test_pre.reshape(y_test_pre.shape[0], 1)

    print('==================================================================')
    print('PU-Train-SOM: ', purity_measure(clusters=y_train_pre, classes=trainY))
    print('PU-Valid-SOM: ', purity_measure(clusters=y_valid_pre, classes=validY))
    print('PU-Test-SOM: ', purity_measure(clusters=y_test_pre, classes=testY))

    print('RI-Train-SOM: ', rand_index(clusters=y_train_pre, classes=trainY))
    print('RI-Valid-SOM: ', rand_index(clusters=y_valid_pre, classes=validY))
    print('RI-Test-SOM: ', rand_index(clusters=y_test_pre, classes=testY))

    print('F-Measure-Train-SOM: ', f_measure(clusters=y_train_pre, classes=trainY))
    print('F-Measure-Valid-SOM: ', f_measure(clusters=y_valid_pre, classes=validY))
    print('F-Measure-Test-SOM: ', f_measure(clusters=y_test_pre, classes=testY))