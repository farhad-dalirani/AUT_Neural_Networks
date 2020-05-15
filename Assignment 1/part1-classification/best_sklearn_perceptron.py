"""
This code finds good parameters for achieving excellent achievement on
perceptrons (one vs all)
"""

if __name__ == '__main__':
    import numpy as np
    from read_iris import read_iris_dataset
    from k_fold import k_fold_cross_validation
    from sklearn.linear_model import Perceptron
    from sklearn.model_selection import KFold

    order = '1'
    max_epochs = 200
    K_fold = 10
    best_weights = None
    kf = KFold(n_splits=10)

    # write in file
    file = open('find_good_sklearn_perceptrons.txt', 'w')
    ratio_of_train_set = 0.7
    ratio_of_test_set = 0.15
    ratio_of_valid_set = 0.15
    min_test_error = 10000.0
    train_error_of_best = 10000.0
    k_fold_error_of_best = -1
    best_learning = np.inf
    best_random_state=np.inf

    for random_state in range(0, 10):

        # read train and test from iris dataset
        trainX, validX, testX, trainY, validY, testY = read_iris_dataset(rand_state=random_state,
                                                         train_ratio=ratio_of_train_set,
                                                         test_ratio=ratio_of_test_set,
                                                         valid_ratio=1-(ratio_of_train_set+ratio_of_valid_set),
                                                         order=order)
        # print(trainX, testX, trainY, testY)

        # map labels name
        map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
        temp = np.zeros(trainY.shape, dtype=int)
        for idx, label in enumerate(trainY):
            temp[idx] = map[trainY[idx][0]]
        trainY = np.copy(temp)

        temp = np.zeros(testY.shape, dtype=int)
        for idx, label in enumerate(testY):
            temp[idx] = map[testY[idx][0]]
        testY = np.copy(temp)

        for learning_rate in np.arange(0.01, 1, 0.1):

            # train perceptrons (all vs on)
            units = Perceptron(eta0=learning_rate, random_state=random_state, max_iter=max_epochs)
            k_fold_units = Perceptron(eta0=learning_rate, random_state=random_state, max_iter=max_epochs)
            # train perceptrons
            units.fit(X=trainX, y=trainY)

            # calculate K-Fold Error
            train_arguments = {}

            #k_fold_error = k_fold_cross_validation(learner=k_fold_units, argument_of_learner=train_arguments,
            #                                       X=np.copy(trainX), y=np.copy(trainY), k=K_fold)
            #print('kkkkkkkkkkkkkkkkkkkkkkkkk')

            file.write('\nVV======================VVV=======================VV')
            print('===================================================V')
            file.write('\nlearning_rate: {}'.format(learning_rate))
            print('learning_rate: {}'.format(learning_rate))
            file.write('\nrandom state: {}'.format(random_state))
            print('random state: {}'.format(random_state))
            file.write('\nratio of train set: {}'.format(ratio_of_train_set))
            print('ratio of train set: {}'.format(ratio_of_train_set))

            # calculate K-fold error
            k_fold_error = 0
            for train_index, test_index in kf.split(trainX):
                #print("TRAIN:", train_index, "TEST:", test_index)
                X_train, X_test = trainX[train_index], trainX[test_index]
                y_train, y_test = trainY[train_index], trainY[test_index]

                k_fold_units.fit(X_train, y_train)

                # error of sklearn perceptrons(one vs all) for training set
                pred = []
                for x in X_test:
                    prediction = units.predict([x])
                    # print(prediction)
                    pred.append(prediction)

                fold_error = 0
                for _ in range(0, len(pred)):
                    if pred[_] != y_test[_]:
                        fold_error += 1

                k_fold_error += fold_error/len(pred)
            k_fold_error /= 10

            # error of perceptrons(one vs all) for training set
            pred = []
            for x in trainX:
                prediction = units.predict([x])
                # print(prediction)
                pred.append(prediction)

            train_error = 0
            for _ in range(0, len(pred)):
                if pred[_] != trainY[_]:
                    train_error += 1

            print('10-fold-error: {}'.format(k_fold_error))
            file.write('\n10-fold-error error: {}'.format(k_fold_error))

            file.write('\nTrain Error(Ordinary Error, not k-fold)= {0}'.format(train_error / len(pred)))
            print('Train Error(Ordinary Error, not k-fold)= {0}'.format(train_error / len(pred)))
            et = train_error / len(pred)

            # error of perceptrons(one vs all) for test set
            pred = []
            for x in testX:
                prediction = units.predict([x])
                pred.append(prediction)

            test_error = 0
            for _ in range(0, len(pred)):
                if pred[_] != testY[_]:
                    test_error += 1

            file.write('\nTest Error= {0}'.format(test_error / len(pred)))
            print('Test Error= {0}'.format(test_error / len(pred)))

            if min_test_error > (test_error / len(pred)):
                min_test_error = test_error / len(pred)
                train_error_of_best = et
                best_random_state = random_state
                best_learning = learning_rate
                best_weights = units.coef_
                k_fold_error_of_best = k_fold_error

            file.write('\n===================================================^\n')
            print('===================================================^')

    print('\n********************************')
    file.write('\n********************************')
    print('train error of best test error: {}'.format(train_error_of_best))
    file.write('\nmin test error: {}'.format(min_test_error))
    print('10-fold-error of best test error: {}'.format(k_fold_error))
    file.write('\n10-fold-error error of best test error: {}'.format(k_fold_error))
    print('min test error: {}'.format(min_test_error))
    file.write('\ntrain error of best test error: {}'.format(train_error_of_best))
    print('best learning rate: {}'.format(best_learning))
    file.write('\nbest learning rate: {}'.format(best_learning))
    print('best random state: {}'.format(best_random_state))
    file.write('\nbest random state: {}'.format(best_random_state))
    print('best weights: {}'.format(best_weights))
    file.write('\nbest weights: {}'.format(best_weights))

    file.close()