if __name__ == '__main__':

    # Import packages
    import numpy as np
    from activation_functions import activation, activation_derivative
    import time
    import copy
    import sys
    from read_dataset import read_digit, read_digit_letter
    import json
    from mlp import MLP
    import threading
    from plotting import plotting
    from k_fold import k_fold_cross_validation

    # get inputs
    print('Do want to give parameters for training a MLP via a file or you want to five a trained MLP that stored in file:')
    print('>    1. I want to give parameters via file')
    print('>    2. I want to give a trained MLP via file')
    train_load = int(float(input('Enter 1 or 2: ')))

    if train_load != 1 and train_load != 2:
        raise ValueError('Input should be 1 1 or two!')

    K_fold = None
    if train_load == 1:
        # read parameter via file
        path_config = str(input('\nEnter name of the config file that contains necessary parameters for training MLP.'+
                                '\n(It should be located in directory of code. Example: input.json) : '))
        # read json file
        try:
            data = json.load(open(path_config))
        except:
            print('Can\'t open input json file!')
            exit(2)

        digits_leters_digits = str(data['digits_letters_digits']) # a string 'digits' or 'letters_digits'
        ratio_of_train_set = float(data['ratio_of_train_set']) # (0.0, 1.0)
        ratio_of_valid_set = float(data['ratio_of_valid_set']) # (0.0, 1.0)
        ratio_of_test_set = float(data['ratio_of_test_set']) # (0.0, 1.0)
        learning_rate = float(data['learning_rate']) # positive number
        max_epochs = int(float(data['max_epochs'])) # integer number
        cut_cost = float(data['cut_cost']) # positive float number for stop training when cost less than it
        if cut_cost == -1:
            cut_cost = None
        random_state = int(float(data['random_state'])) # integer number
        K_fold = int(float(data['K_fold'])) # integer number
        if K_fold == -1:
            K_fold = None
        neurons_layes = data['neurons_layes'] # a list [number of feature, neurons in layer1,...,
        # neurons in layerN-1, neurons in output Layer]
        type_layers = data['type_layers'] # a list of activation type in each layer like ['tanh',...,'tanh', 'sigmoid']
        type_of_cost = str(data['type_of_cost'])  # a string 'MSE' or 'cross_entropy'
        mode = str(data['mode']) # a string 'batch', 'stochastic', 'momentum', 'steepest'

    else:
        path_trained_mlp = str(input('\nEnter name of the json file that contains a trained MLP.' +
                                     '\n(It should be located in directory of code. Example: input.json) : '))
        digits_leters_digits = int(float(input('Do you want to use digit dataset of digit-letter dataset?'+
                                         '\n>   Enter 1 if you want to use digit dataset'+
                                         '\n>   Enter 2 if you want to use digit-letter dataset: ')))
        if digits_leters_digits == 1:
            digits_leters_digits = 'digits'
        elif digits_leters_digits == 2:
            digits_leters_digits = 'letters_digits'
        else:
            raise ValueError('Input should be 1 or 2!')

        ratio_of_train_set = float(input('What ratio of dataset should be used for training(0.0, 1.0): ')) # (0.0, 1.0)
        ratio_of_valid_set = float(input('What ratio of dataset should be used for validation(0.0, 1.0): ')) # (0.0, 1.0)
        ratio_of_test_set = float(input('What ratio of dataset should be used for testing(0.0, 1.0): ')) # (0.0, 1.0)
        random_state = int(float(input('Enter an integer for seeding random numbers: ')))

    if digits_leters_digits == 'digits':
        # read dataset-only digits
        trainX, validX, testX, trainY, validY, testY = read_digit(train_ratio=ratio_of_train_set, valid_ratio=ratio_of_valid_set,
                                                                  test_ratio=ratio_of_test_set, random_state=random_state)
    else:
        # read dataset- digits and letters
        trainX, validX, testX, trainY, validY, testY = read_digit_letter(train_ratio=0.80, valid_ratio=0.10,
                                                                         test_ratio=0.10, random_state=random_state)
    # transpose
    trainX = trainX.T
    validX = validX.T
    testX = testX.T
    trainY = trainY.T
    validY = validY.T
    testY = testY.T

    # create a thread for calculating k-fold
    k_f_thread = None
    if K_fold is not None and train_load == 1:
        print('Calculating {}-fold in a separate thread has been began.'.format(K_fold))
        mlp_k_fold = MLP(n=neurons_layes, activations=type_layers, type_of_cost=type_of_cost, learning_rate=learning_rate,
                  max_epoch=max_epochs, mode=mode, random_state=random_state, plot_info=False, cut_cost=cut_cost, save=False)
        arg = {}
        X_ = np.copy(trainX)
        Y_ = np.copy(trainY)
        k_f_thread = threading.Thread(target=k_fold_cross_validation, args=[mlp_k_fold, arg, X_, Y_, K_fold])
        k_f_thread.daemon = True
        k_f_thread.start()

    plot_thread = None
    if train_load == 1:

        # create a MLP network
        mlp = MLP(n=neurons_layes, activations=type_layers, type_of_cost=type_of_cost, learning_rate=learning_rate,
                  max_epoch=max_epochs, mode=mode, random_state=random_state, plot_info=True, cut_cost=cut_cost)

        # create a thread for plotting
        try:
            plot_thread = threading.Thread(target=plotting, args=[mlp, trainX, trainY, validX, validY])
            plot_thread.daemon = True
            plot_thread.start()
        except:
            print('error happened')

        # train MLP
        mlp.fit(X=trainX, y=trainY)
    else:
        # load weights and bias and other necessary info from file
        # create a MLP network
        mlp = MLP(n=[], activations=[], random_state=random_state)
        mlp.load(path_trained_mlp)

    print('======================V=======================')
    # Calculate errors
    trainY_hat = mlp.predict(trainX)
    train_error = np.sum(np.abs(trainY-trainY_hat))/(trainY.shape[1]*2)
    print("Train Error: {}".format(train_error))

    validY_hat = mlp.predict(validX)
    valid_error = np.sum(np.abs(validY - validY_hat)) / (validY.shape[1] * 2)
    print("Validation Error: {}".format(valid_error))

    testY_hat = mlp.predict(testX)
    test_error = np.sum(np.abs(testY - testY_hat)) / (testY.shape[1] * 2)
    print("Test Error: {}".format(test_error))

    if train_load == 2:
        total_error_dataset = (train_error*trainY.shape[1]+
                               test_error*testY.shape[1]+
                               valid_error * validY.shape[1])/(trainY.shape[1]+testY.shape[1]+validY.shape[1])
        print("\n>    Total Error on dataset: {}\n".format(total_error_dataset))

    if k_f_thread is not None:
        k_f_thread.join()
    print('======================^=======================')

    # wait for plotting thread
    if plot_thread is not None:
        plot_thread.join()

