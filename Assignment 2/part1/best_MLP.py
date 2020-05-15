
if __name__ == '__main__':

    # Import packages
    import numpy as np
    import copy
    from read_dataset import read_digit, read_digit_letter
    import json
    from mlp import MLP
    from plotting import plotting

    digits_leters_digits = 'digits'
    # digits_leters_digits = 'letters_digits'

    ratio_of_train_set = 0.80
    ratio_of_valid_set = 0.10
    ratio_of_test_set = 0.10
    random_state = 0

    if digits_leters_digits == 'digits':
        # read dataset-only digits
        trainX, validX, testX, trainY, validY, testY = read_digit(train_ratio=ratio_of_train_set,
                                                                  valid_ratio=ratio_of_valid_set,
                                                                  test_ratio=ratio_of_test_set,
                                                                  random_state=random_state)
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

    #type_layers = ['tanh', 'sigmoid']
    #neurons_layes = [1024, 80, 10]

    #type_layers = ['tanh', 'tanh', 'tanh', 'sigmoid']
    #neurons_layes = [1024, 10, 10, 10, 10]

    #type_layers = ['tanh', 'tanh', 'tanh','tanh', 'tanh', 'tanh', 'sigmoid']
    #neurons_layes = [1024, 20, 20, 20, 20, 20, 20, 10]

    type_layers = ['tanh', 'sigmoid']
    neurons_layes = [1024, 64, 10]

    type_of_cost = 'MSE'
    #mode = 'momentum'
    #mode = 'batch'
    mode = 'steepest'
    cut_cost = None

    res_file = open('result_'+mode+'_'+type_of_cost+'_layers '+str(neurons_layes)+'_'+digits_leters_digits+'.dat','w')
    # write in file
    res_file.write('Dataset: \n')
    res_file.write(digits_leters_digits)
    res_file.write('\nNumber of neurons in each layer:\n')
    res_file.write(str(neurons_layes))
    res_file.write('\nActivation of each layer:\n')
    res_file.write(str(type_layers))
    res_file.write('\nCost type:\n')
    res_file.write(type_of_cost)
    res_file.write('\nMode of training:\n')
    res_file.write(mode)
    res_file.write('\n==============================================\n')

    min_test_error = np.inf
    associated_valid_error = None
    associated_train_error = None
    associated_max_epoch = None
    associated_learning_rate = None

    for max_epochs in [5, 10, 20]:
    #for max_epochs in [5, 50, 100, 1000, 1000]:
        for learning_rate in [0.001, 0.003, 0.006, 0.01, 0.06, 0.1, 0.5]:
            # create a MLP network
            mlp = MLP(n=neurons_layes, activations=type_layers, type_of_cost=type_of_cost, learning_rate=learning_rate,
                      max_epoch=max_epochs, mode=mode, random_state=random_state, plot_info=False, cut_cost=cut_cost, save=False)

            # train MLP
            mlp.fit(X=trainX, y=trainY)

            # Calculate errors
            trainY_hat = mlp.predict(trainX)
            train_error = np.sum(np.abs(trainY - trainY_hat)) / (trainY.shape[1] * 2)
            #print("Train Error: {}".format(train_error))

            validY_hat = mlp.predict(validX)
            valid_error = np.sum(np.abs(validY - validY_hat)) / (validY.shape[1] * 2)
            #print("Validation Error: {}".format(valid_error))

            testY_hat = mlp.predict(testX)
            test_error = np.sum(np.abs(testY - testY_hat)) / (testY.shape[1] * 2)
            #print("Test Error: {}".format(test_error))

            # keep parameters of best experiment that has the lowest
            # test error
            if test_error < min_test_error:
                min_test_error = test_error
                associated_train_error = train_error
                associated_valid_error = valid_error
                associated_learning_rate = learning_rate
                associated_max_epoch = max_epochs

            print('======================V=======================')
            print('Max Epoch:')
            print(str(max_epochs))
            print('Learning rate:')
            print(str(learning_rate))
            print('Train Error:')
            print(str(train_error))
            print('Valid Error:')
            print(str(valid_error))
            print('Test Error:')
            print(str(test_error))

            # write in file
            res_file.write('\n======================V=======================\n')
            res_file.write('\nMax Epoch:\n')
            res_file.write(str(max_epochs))
            res_file.write('\nLearning rate:\n')
            res_file.write(str(learning_rate))
            res_file.write('\nTrain Error:\n')
            res_file.write(str(train_error))
            res_file.write('\nValid Error:\n')
            res_file.write(str(valid_error))
            res_file.write('\nTest Error:\n')
            res_file.write(str(test_error))

    # best experiment
    print('======================V=======================')
    print('Best Test Error:')
    print(min_test_error)
    print('Train Error of Best Test Error:')
    print(associated_train_error)
    print('Validation Error of Best Test Error:')
    print(associated_valid_error)
    print('Learning Rate of Best Test Error:')
    print(associated_learning_rate)
    print('Max Epoch of Best Test Error:')
    print(associated_max_epoch)

    res_file.write('\n==============================================\n')
    res_file.write('\nBest Test Error:\n')
    res_file.write(str(min_test_error))
    res_file.write('\nTrain Error of Best Test Error:\n')
    res_file.write(str(associated_train_error))
    res_file.write('\nValidation Error of Best Test Error:\n')
    res_file.write(str(associated_valid_error))
    res_file.write('\nLearning Rate of Best Test Error:\n')
    res_file.write(str(associated_learning_rate))
    res_file.write('\nMax Epoch of Best Test Error:\n')
    res_file.write(str(associated_max_epoch))

    res_file.close()