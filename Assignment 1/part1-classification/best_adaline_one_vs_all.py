"""
This code finds good parameters for achieving excellent achievement on
adalines (one vs all)
"""

if __name__ == '__main__':
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from read_iris import read_iris_dataset
    from adaline_one_vs_all import adaline_one_vs_all
    import json
    from k_fold import k_fold_cross_validation
    from sklearn import preprocessing

    import sys
    import warnings

    order = '1'
    cut_error = None
    max_epochs = 100
    K_fold = 10

    # write in file
    file = open('find_good_adalines.txt', 'w')
    ratio_of_train_set = 0.7
    ratio_of_valid_set = 0.15
    ratio_of_test_set = 0.15
    min_test_error = 10000.0
    best_learning = np.inf
    best_random_state = np.inf

    for random_state in range(0, 8):

        # read train and test from iris dataset
        trainX, validX, testX, trainY, validY, testY = read_iris_dataset(rand_state=random_state,
                                                                         train_ratio=ratio_of_train_set,
                                                                         valid_ratio=ratio_of_valid_set,
                                                                         test_ratio=1 - (ratio_of_train_set + ratio_of_valid_set),
                                                                         order=order)
        # print(trainX, testX, trainY, testY)

        # standard scaler
        scaler = preprocessing.StandardScaler().fit(trainX)
        trainX = scaler.transform(trainX)
        validX = scaler.transform(validX)
        testX = scaler.transform(testX)

        # map labels name
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

        # 0.001, 0.02, 0.002
        for learning_rate in np.arange(0.001, 0.02, 0.002):

            # train adalines (all vs on)
            units = adaline_one_vs_all(number_classes=3, learning_rate=learning_rate, max_epoch=max_epochs,
                                          cut_error=cut_error)
            k_fold_units = adaline_one_vs_all(number_classes=3, learning_rate=learning_rate, max_epoch=max_epochs,
                                                 cut_error=cut_error)

            # train adaline
            units.fit(X=trainX, y=trainY, validX=validX, validY=validY,
                      classes_name=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                      rand_state=random_state, plotting=False, write_in_file=False)

            # calculate K-Fold Error
            train_arguments = {'classes_name': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                               'rand_state': random_state, 'plotting': False, 'write_in_file': False}

            k_fold_error = k_fold_cross_validation(learner=k_fold_units, argument_of_learner=train_arguments,
                                                   X=np.copy(trainX), y=np.copy(trainY), k=K_fold)

            file.write('\nVV======================VVV=======================VV')
            print('===================================================V')
            file.write('\nlearning_rate: {}'.format(learning_rate))
            print('learning_rate: {}'.format(learning_rate))
            file.write('\nrandom state: {}'.format(random_state))
            print('random state: {}'.format(random_state))
            file.write('\nratio of train set: {}'.format(ratio_of_train_set))
            print('ratio of train set: {}'.format(ratio_of_train_set))

            # error of adalines(one vs all) for training set
            pred = []
            for x in trainX:
                prediction = units.predict(x)
                # print(prediction)
                pred.append(prediction)

            train_error = 0
            for _ in range(0, len(pred)):
                if pred[_] != trainY[_]:
                    train_error += 1

            file.write('\n{0}-fold-cross-validation error is: {1}'.format(K_fold, k_fold_error))
            print('{0}-fold-cross-validation error is: {1}'.format(K_fold, k_fold_error))
            file.write('\nTrain Error(Ordinary Error, not k-fold)= {0}'.format(train_error / len(pred)))
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
            #print(pred)

            file.write('\nTest Error= {0}'.format(test_error / len(pred)))
            print('Test Error= {0}'.format(test_error / len(pred)))

            if min_test_error > (test_error / len(pred)):
                min_test_error = test_error / len(pred)
                best_random_state = random_state
                best_learning = learning_rate

            file.write('\n===================================================^\n')
            print('===================================================^')

    print('\n********************************')
    file.write('\n********************************')
    print('min test error: {}'.format(min_test_error))
    file.write('\nmin test error: {}'.format(min_test_error))
    print('best learning rate: {}'.format(best_learning))
    file.write('\nbest learning rate: {}'.format(best_learning))
    print('best random state: {}'.format(best_random_state))
    file.write('\nbest random state: {}'.format(best_random_state))

    file.close()