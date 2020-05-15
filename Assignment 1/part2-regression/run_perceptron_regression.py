"""
This python code use 'multi class perceptron_one_vs_all' class for classfying iris dataset.
It gets inputs from standard input and file.
"""

if __name__ == '__main__':

    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from read_auto_mpg_dataset import read_auto_mpg_dataset, read_auto_mpg_dataset_continuous
    from perceptron_regression import regression_perceptron
    import json
    from sklearn import preprocessing

    import sys
    import warnings

    # turn off warning like future python version warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    do_train = True
    cut_error = None

    all_features = 1
    order = 1

    # determine where program should get parameters
    key_file = int(input('How do you want to give learning parameters to the program?'+ '\n'+
          '>    1: I give parameters via Standard Input / I have trained regression perceptrons in a file'+ '\n'+
          '>    2: I give parameters via a file'+ '\n'+
          'Input 1 or 2: '))

    if isinstance(key_file, str) == False:
        key_file = str(key_file)

    if key_file == '1':

        # determine user wants train regression perceptrons or read trained regression perceptrons from file
        train_yes_no = int(input('Do you want to train regression Perceptrons(one vs all) or read trained regression perceptrons(weights) from file?'+ '\n'+
          '>    1: I want to give parameters and train regression Perceptrons'+ '\n'+
          '>    2: Read trained regression perceptrons from file'+ '\n'+
          'Input 1 or 2: '))

        if isinstance(train_yes_no, str) == False:
            train_yes_no = str(train_yes_no)

        if train_yes_no == '1':
            all_features = 1
            all_features = int(input('Do you want to use all features or just continuous features?' + '\n' +
                                     '>    1: Use all feature for training' + '\n' +
                                     '>    2: Just use continuous features' + '\n' +
                                     'Input 1 or 2: '))
            order = '1'
            order = str(input('Do you want to use first order features or second order features?' + '\n' +
                              '>    1: First order features' + '\n' +
                              '>    2: Second order features' + '\n' +
                              'Input 1 or 2: '))

            ratio_of_train_set = float(input('\n >How much of data set should be used for training?'+'\n'
                                             +'A number in interval (0.0, 1.0): '))
            ratio_of_valid_set = float(input('\n >How much of data set should be used for validation?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            ratio_of_test_set = float(input('\n >How much of data set should be used for test?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            learning_rate = float(input('\n> Learning Rate: '))
            max_epochs = int(float(input('\n> Maximum number of epochs(Integer): ')))
            cut_error = float(input('\n> Do you want regression perceptrons stop training when error during training falls under a threshold:'+'\n'+
                                        '>  Enter -1 if you don\'t want error limit and maximum epochs is enough'+'\n'+
                                     '>  Otherwise, Enter a value in interval (0.0, 1.0) for cut off error'+'\n'+
                                     'Enter -1 or a number in interval(0.0, 1.0): '))
            if cut_error == -1.0:
                cut_error = None

            random_state = int(float(input('\n> Enter an integer, it is used as seed for generating random numbers and'+'\n'
                                       +' repetitain of experiment:')))

        elif train_yes_no == '2':
            do_train = False
            # read necessary input and parameters from standard input

            all_features = 1
            all_features = int(input('Do you want to use all features or just continuous features?' + '\n' +
                                     '>    1: Use all feature for training' + '\n' +
                                     '>    2: Just use continuous features' + '\n' +
                                     'Input 1 or 2: '))

            order = '1'
            order = str(input('Do you want to use first order features or second order features?' + '\n' +
                              '>    1: First order features' + '\n' +
                              '>    2: Second order features' + '\n' +
                              'Input 1 or 2: '))

            ratio_of_train_set = float(input('\n >How much of data set should be used for training?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            ratio_of_valid_set = float(input('\n >How much of data set should be used for validation?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            ratio_of_test_set = float(input('\n >How much of data set should be used for test?' + '\n'
                                            + 'A number in interval (0.0, 1.0): '))
            random_state = int(
                float(input('\n> Enter an integer, it is used as seed for generating random numbers and' + '\n'
                            + ' repetitain of experiment:')))
            # name of jason file which trained regression perceptrons are there
            file_name = input(
                '\n> Enter name of json file that contains weights of trained regression perceptros'+ '\n'+
                '(shoud be in directory of code):')

        else:
            raise ValueError('Input should be 1 or 2.')



    elif key_file == '2':

        file_name = input('> Enter name of json file that contains parameters \nlike learning rata and etc.(shoud be in directory of code):')
        if isinstance(file_name, str) == False:
            file_name = str(file_name)

        # read parameters for learning from json file
        with open(file_name) as json_data_file:
            data = json.load(json_data_file)
        print('parameters:\n', data, '\n')
        all_features = float(data['all_features'])
        order = str(data['order'])
        ratio_of_train_set = float(data['ratio_of_train_set'])
        ratio_of_valid_set = float(data['ratio_of_valid_set'])
        ratio_of_test_set = float(data['ratio_of_test_set'])
        learning_rate = float(data['learning_rate'])
        max_epochs = int(float(data['max_epochs']))
        cut_error = float(data['cut_error'])
        if cut_error == -1.0:
            cut_error = None
        random_state = int(float(data['random_state']))

    else:
        raise ValueError('Input should be 1 or 2.')

    if all_features == 1:
        # read all features
        # read train and test from iris dataset
        trainX, validX, testX, trainY, validY, testY = read_auto_mpg_dataset(rand_state=random_state,
                                                                    train_ratio=ratio_of_train_set,
                                                                    valid_ratio=ratio_of_valid_set,
                                                                    test_ratio=1 - (ratio_of_train_set + ratio_of_valid_set),
                                                                    order=order)
        weight_index_feature=3
    else:
        # read jost continous features
        trainX, validX, testX, trainY, validY, testY = read_auto_mpg_dataset_continuous(rand_state=random_state,
                                                                             train_ratio=ratio_of_train_set,
                                                                             valid_ratio=ratio_of_valid_set,
                                                                             test_ratio=1 - (
                                                                             ratio_of_train_set + ratio_of_valid_set),
                                                                             order=order)
        weight_index_feature = 2

    #print(trainX, testX, trainY, testY)

    # standard scaler
    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)
    validX = scaler.transform(validX)
    testX = scaler.transform(testX)


    # train regression perceptrons ()
    if do_train == True:
        unit = regression_perceptron(learning_rate=learning_rate, max_epoch=max_epochs, cut_error=cut_error)
    else:
        unit = regression_perceptron()

    if do_train == True:

        # train regression perceptrons
        unit.fit(X=trainX, y=trainY, validX=validX, validY=validY,
                  rand_state=random_state, plotting=True, write_in_file=True, feature_i=weight_index_feature)

    else:
        # load regression perceptros from file
        unit.load_regression_perceptron_from_file(file_name)

        # plot mpg-weight
        plt.figure()
        # plt.legend(loc=3)
        plt.scatter(x=trainX[:, weight_index_feature], y=trainY, c='red')
        weight_x = np.linspace(start=-2, stop=2, num=100)
        _x = np.zeros((100, trainX.shape[1]))
        _x[:, weight_index_feature] = weight_x
        plt.plot(weight_x, [unit.predict(xi) for xi in _x], 'b--')
        plt.pause(0.001)

    if do_train == True:
        train_error = unit.get_error_in_epochs()
        train_error = train_error[len(train_error) - 1]
        print('Train MSE Error= ', train_error)

        validation_error = unit.get_valid_error_in_epochs()
        if len(validation_error) > 0:
            validation_error = validation_error[len(validation_error) - 1]
            print('Validation MSE-Error= ', validation_error)

    pred = np.zeros(testY.shape)
    for idx, x in enumerate(testX):
        pred[idx] = unit.predict(x)

    # print(testY)
    # print(pred)
    # for x,y in zip(testY, pred):
    #    print(x,y)

    test_error = 0
    for _ in range(0, pred.shape[0]):
        test_error += (pred[_] - testY[_]) ** 2
    print('Test MSE-Error= ', test_error / pred.shape[0])
    plt.show(block=True)

    plt.show(block=True)

    print('\n======================')
    input('For closing window press any key...')