def k_fold_cross_validation(learner, argument_of_learner, X_, y_, k):
    """
    This function uses k-fold-cv to evaluate learner, learner can be
    multi-layer perceptron, perceptron, adaline and ....

    :param learner: Is an instance of multi-layer perceptron, perceptron or adaline or ... that learns
            from data to predict label of new inputs.

    :param argument_of_learner: is a list
        that contains necessary argument of
        learningFunction.
    :param X: training data
    :param y: labels
    :param K: number of folds
    :return: return average k-fold cv error
    """
    from math import floor
    import copy
    import numpy as np

    X = X_.T
    y = y_.T

    # average error on 10 folds
    average_error = 0

    # calculate size of each fold
    fold_size = floor(len(X) / k)

    # A list that contain k folds
    foldsX = []
    foldsY = []

    # Divide dataSet to k fold
    for fold in range(k-1):
        foldsX.append(X[fold * fold_size:(fold + 1) * fold_size])
        foldsY.append(y[fold * fold_size:(fold + 1) * fold_size])

    foldsX.append(X[(k - 1) * fold_size::])
    foldsY.append(y[(k - 1) * fold_size::])

    # Train and test learning function with k different forms
    for index1, i in enumerate(foldsX):
        # Test contains fold[i]
        testX = np.copy(i)
        testY = np.copy(foldsY[index1])

        # Train contains all folds except fold[i]
        trainX = np.array([])
        trainY = np.array([])
        for index2, j in enumerate(foldsX):
            if index2 != index1:
                if trainX.shape[0] == 0:
                    trainX = np.copy(j)
                    trainY = np.copy(foldsY[index2])
                else:
                    trainX = np.vstack((trainX, j))
                    trainY = np.vstack((trainY, foldsY[index2]))

        # train learner
        arguments = copy.deepcopy(argument_of_learner)
        arguments['X'] = copy.deepcopy(trainX.T)
        arguments['y'] = copy.deepcopy(trainY.T)
        learner.fit(**arguments)

        # Evaluate performance of learningFunction
        testY_hat = learner.predict(testX.T)
        test_error = np.sum(np.abs(testY.T - testY_hat)) / (testY_hat.shape[1] * 2)

        #print('Fold {0}, Error is {1}'.format(index1, fold_error / testX.shape[0]))
        average_error += test_error


    average_error /= k

    print('\n{}-fold error is: {}\n'.format(k, average_error))

    return average_error
