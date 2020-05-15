"""
This python code use 'adaline_one_vs_all' class for classfying iris dataset.
It gets inputs from standard input and file.
"""

if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from read_iris import read_iris_dataset
    from adaline_one_vs_all import adaline_one_vs_all
    import json
    from k_fold import k_fold_cross_validation
    from sklearn import preprocessing
    import sys
    import warnings

    # turn off warning like future python version warnings
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    do_train = True
    order = '1'
    cut_error = None

    # determine where program should get parameters
    key_file = int(input('How do you want to give learning parameters to the program?'+ '\n'+
          '>    1: I give parameters via Standard Input / I have trained adalines in a file'+ '\n'+
          '>    2: I give parameters via a file'+ '\n'+
          'Input 1 or 2: '))

    if isinstance(key_file, str) == False:
        key_file = str(key_file)

    if key_file == '1':

        # determine user wants train adalines or read trained adalines from file
        train_yes_no = int(input('Do you want to train adalines(one vs all) or read trained adalines(weights) from file?'+ '\n'+
          '>    1: I want to give parameters and train Adaloines'+ '\n'+
          '>    2: Read trained adalines from file'+ '\n'+
          'Input 1 or 2: '))

        if isinstance(train_yes_no, str) == False:
            train_yes_no = str(train_yes_no)

        if train_yes_no == '1':

            # read necessary input and parameters from standard input
            order = str(input('\nDo you want to use first order or second order adalines ?' + '\n'
                                             +  '>   Enter 1 for first order ' + '\n'+
                                                '>   Enter 2 for second order'+'\n'+'Enter 1 or 2:'))
            ratio_of_train_set = float(input('\n >How much of data set should be used for training?'+'\n'
                                             +'A number in interval (0.0, 1.0): '))
            ratio_of_valid_set = float(input('\n >How much of data set should be used for validation?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            ratio_of_test_set = float(input('\n >How much of data set should be used for test?' + '\n'
                                            + 'A number in interval (0.0, 1.0): '))
            learning_rate = float(input('\n> Learning Rate: '))
            max_epochs = int(float(input('\n> Maximum number of epochs(Integer): ')))
            cut_error = float(input('\n> Do you want adaliness stop training when MSE error during training falls under a threshold:'+'\n'+
                                        '>  Enter -1 if you don\'t want MSE error limit and maximum epochs is enough'+'\n'+
                                     '>  Otherwise, Enter a value in interval (0.0, 1.0) for cut off MSE error'+'\n'+
                                     'Enter -1 or a number in interval(0.0, 1.0): '))
            if cut_error == -1.0:
                cut_error = None

            random_state = int(float(input('\n> Enter an integer, it is used as seed for generating random numbers and'+'\n'
                                       +' repetitain of experiment:')))
            K_fold = int(float(input('\n> Enter an integer that determines number of fold for K-Fold-Cross Validation:')))

        elif train_yes_no == '2':
            do_train = False
            # read necessary input and parameters from standard input
            order = str(input('\n >Trained adalines in file are first order or second order?' + '\n'
                              + '    Enter 1 for first order ' + '\n' +
                              '   Enter 2 for second order'+'\n'+'Enter 1 or 2:'))

            ratio_of_train_set = float(input('\n >How much of data set should be used for training?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            ratio_of_valid_set = float(input('\n >How much of data set should be used for validation?' + '\n'
                                             + 'A number in interval (0.0, 1.0): '))
            ratio_of_test_set = float(input('\n >How much of data set should be used for test?' + '\n'
                                            + 'A number in interval (0.0, 1.0): '))
            random_state = int(
                float(input('\n> Enter an integer, it is used as seed for generating random numbers and' + '\n'
                            + ' repetitain of experiment:')))
            # name of jason file which trained adalines are there
            file_name = input(
                '\n> Enter name of json file that contains weights of trained adalines'+ '\n'+
                '(should be in directory of code):')

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
        ratio_of_train_set = float(data['ratio_of_train_set'])
        ratio_of_valid_set = float(data['ratio_of_valid_set'])
        ratio_of_test_set = float(data['ratio_of_test_set'])
        learning_rate = float(data['learning_rate'])
        max_epochs = int(float(data['max_epochs']))
        cut_error = float(data['cut_error'])
        if cut_error == -1.0:
            cut_error = None
        random_state = int(float(data['random_state']))
        K_fold = int(float(data['K_fold']))
        order = str(data['order'])

    else:
        raise ValueError('Input should be 1 or 2.')

    # read train and test from iris dataset
    trainX, validX, testX, trainY, validY, testY = read_iris_dataset(rand_state=random_state,
                                                                    train_ratio=ratio_of_train_set,
                                                                    valid_ratio=ratio_of_valid_set,
                                                                    test_ratio=1 - (ratio_of_train_set + ratio_of_valid_set),
                                                                    order=order)

    # standard scaler
    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)
    validX = scaler.transform(validX)
    testX = scaler.transform(testX)

    #print(trainX, testX, trainY, testY)

    #map labels name
    map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    temp = np.zeros(trainY.shape, dtype=int)
    for idx, label in enumerate(trainY):
        temp[idx] = map[trainY[idx][0]]
    trainY = np.copy(temp)

    temp = np.zeros(validY.shape, dtype=int)
    for idx, label in enumerate(validY):
        temp[idx] = map[validY[idx][0]]
    validY = np.copy(temp)

    temp = np.zeros(testY.shape, dtype=int)
    for idx, label in enumerate(testY):
        temp[idx] = map[testY[idx][0]]
    testY = np.copy(temp)

    # train adalines (all vs on)
    if do_train == True:
        units = adaline_one_vs_all(number_classes=3, learning_rate=learning_rate, max_epoch=max_epochs, cut_error=cut_error)
        k_fold_units = adaline_one_vs_all(number_classes=3, learning_rate=learning_rate, max_epoch=max_epochs, cut_error=cut_error)
    else:
        units = adaline_one_vs_all(number_classes=3)
        k_fold_units = adaline_one_vs_all(number_classes=3)

    if do_train == True:

        # train adalines
        units.fit(X=trainX, y=trainY,  validX=validX, validY=validY,
                  classes_name=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                  rand_state=random_state, plotting=True, write_in_file=True)

        # calculate K-Fold Error
        train_arguments = {'classes_name': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                           'rand_state': random_state, 'plotting': False, 'write_in_file': False}
        print('Calculating {0}-Fold Cross Validation ...'.format(K_fold))
        k_fold_error = k_fold_cross_validation(learner=k_fold_units, argument_of_learner=train_arguments,
                                               X=np.copy(trainX), y=np.copy(trainY), k=K_fold)
        print('{0}-fold-cross-validation error is: {1}'.format(K_fold, k_fold_error))

    else:
        # load adalines from file
        units.load_adalines_from_file(file_name)

    # error of adalines(one vs all) for training set
    pred = []
    for x in trainX:
        prediction = units.predict(x)
        #print(prediction)
        pred.append(prediction)

    train_error = 0
    for _ in range(0, len(pred)):
        if pred[_] != trainY[_]:
            train_error += 1
    print('Train Error(Ordinary Error, not k-fold)= {0}'.format(train_error / len(pred)))

    # error of adalines(one vs all) for test set
    pred = []
    for x in testX:
        prediction = units.predict(x)
        pred.append(prediction)

    test_error = 0
    for _ in range(0, len(pred)):
        if pred[_] != testY[_]:
            test_error += 1
    print('Test Error= {0}'.format(test_error/ len(pred)))

    plt.show(block=True)

    print('\n======================')
    input('For closing window press any key...')