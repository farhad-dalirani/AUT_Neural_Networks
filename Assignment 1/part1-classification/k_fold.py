def k_fold_cross_validation(learner, argument_of_learner, X, y, k):
    """
    This function uses k-fold-cv to evaluate learner, learner can be
    perceptron, adaline and ....

    :param learner: Is an instance of perceptron or adaline or ... that learns
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
        arguments['X'] = copy.deepcopy(trainX)
        arguments['y'] = copy.deepcopy(trainY)
        learner.fit(**arguments)

        # Evaluate performance of learningFunction
        fold_error=0
        for idx, point in enumerate(testX):

            # Find label of sample by learner
            label = learner.predict(point)

            # If it is misclassified add it to error
            if label != testY[idx]:
                fold_error += 1

        #print('Fold {0}, Error is {1}'.format(index1, fold_error / testX.shape[0]))
        average_error += fold_error / testX.shape[0]

    average_error /= k

    return average_error

#####################################################
#                                                   #
#           test part for testing correctness of    #
#                  k_fold function                  #
#                                                   #
#####################################################
if __name__ == '__main__':

    # small example for testing k-fold function
    import numpy as np

    import matplotlib.pyplot as plt
    from read_iris import read_iris_dataset
    from perceptron_one_vs_all import perceptron_one_vs_all

    # read train and test from iris dataset
    trainX, testX, trainY, testY = read_iris_dataset(rand_state=0,
                                                     train_ratio=0.7,
                                                     test_ratio=0.15)
    #print(trainX, testX, trainY, testY)

    #map labels name
    map = {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica': 3}
    temp = np.zeros(trainY.shape, dtype=int)
    for idx, label in enumerate(trainY):
        temp[idx] = map[trainY[idx][0]]
    trainY = np.copy(temp)

    temp = np.zeros(testY.shape, dtype=int)
    for idx, label in enumerate(testY):
        temp[idx] = map[testY[idx][0]]
    testY = np.copy(temp)

    # train perceptrons (all vs on)
    units = perceptron_one_vs_all(number_classes=3,learning_rate=0.01,max_epoch=100)
    #units.fit(trainX, trainY, classes_name=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    #          rand_state=0, plotting=True)

    train_argument = {'classes_name': ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
                        'rand_state': 0, 'plotting': False, 'write_in_file': False}

    k_fold_error = k_fold_cross_validation(learner=units, argument_of_learner=train_argument, X=trainX, y=trainY, k=5)

    print('k-fold-cross-validation error is: ', k_fold_error)
    plt.show(block=True)
    print('======================')
