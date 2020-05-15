from adaline import adaline
import numpy as np
import json

class adaline_one_vs_all:
    """
    This Class classifies multi-class data with adaline by
    using on versus all.

    __number_of_class: number of different classes
    __learning_rate: learning rate of adaline
    __maximum_epoch: maximum number of epochs that adaline is
                        allowed to use for training
    __adalines: adalines for classification
    __cut_error: stop training if error fall under specific value, if it's none don't consider it
    """

    def __init__(self, number_classes, learning_rate=0.1, max_epoch=200, cut_error=None):
        """

        :param number_classes: number of different classes
        :param learning_rate: learning rate of adaline
        :param max_epoch: maximum number of epochs that adaline is
                        allowed to use for training
        :param cut_error: stop training if error fall under specific
                value, if it's none don't consider it
        """
        self.__number_of_class = number_classes
        self.__learning_rate = learning_rate
        self.__maximum_epoch = max_epoch
        self.__adalines = []
        # stop training if error fall under specific value, if it's none don't consider it
        self.__cut_error = cut_error

        # initial 'number_classes' adalines
        for i in range(0, self.__number_of_class):
            self.__adalines.append(adaline(learning_rate=learning_rate, max_epoch=max_epoch,  cut_error=cut_error))

    def fit(self, X, y, validX=None, validY=None, classes_name=None, rand_state=0, plotting=False, write_in_file=False):
        """
        Train adalines by using on vs all. y should contains labels 1,2,..., number_classes
        :param X: Training set features: numpy array of numpy arrays
                    [[feature1 x1, feature2 x2,...,featuren xn],
                    ...,
                    [feature1 xm, feature2 xm,...,featuren xm]]
        :param y: Training set labels: numpy array
                    [[label x1], [label x2], ..., [label xn]]
                    labels shoud be 1,2, ..., number_of_classes
        :param validX: validation set (optional)
        :param validY: labels of validation set (optional)
        :param classes_name: name of classes(optional). ['class1 name',...,'class2 name']
        :param rand_state: for seeding while generating weights
        :param plotting: if it's true, this function plots
                        cost and weights during learning
        :param write_in_file: if true, write weights of adalines in adaline_weights.json
        :return:
        """

        # for each class train a adaline
        for i in range(1, self.__number_of_class+1):

            # if label is equal i set it 1
            # if label of observation is different from class 'i' set it 0
            yi = np.where(y==i, 1, -1)

            # if label is equal i set it 1
            # if label of observation is different from class 'i' set it 0
            validYi = np.where(validY == i, 1, -1)

            # train adaline for class 'i'
            if classes_name == None:
                self.__adalines[i-1].fit(X=X, y=yi, validX=validX, validY=validYi, rand_state=rand_state, plotting=plotting)
            else:
                self.__adalines[i-1].fit(X=X, y=yi, validX=validX, validY=validYi, class_name=classes_name[i-1], rand_state=rand_state, plotting=plotting)

        # if write_in_file true, write weights of perceptrin in a file
        if write_in_file == True:

            weights = []
            for j in range(0, self.__number_of_class):
                weights.append(self.__adalines[j].get_weights().tolist())
            #print('Weights: ', weights)
            with open('output-adaline.json', 'w') as outputfile:
                json.dump(weights, outputfile)

        return self

    def load_adalines_from_file(self, file_name):
        '''
        This function read weights of trained adalines from file
        :param file_name: it's name of a jason file with format like: 'input.json'
                with code should be in same directory
        :return:
        '''
        # read weights from file
        with open(file_name) as json_file:
            weights = json.load(json_file)

        # read adalines from file
        for i in range(0, self.__number_of_class):
            self.__adalines[i].load_weights(weights=weights[i])
        return self

    def predict(self, x):
        """
            This function predict output of adalines(one vs all) for instance X.
            it uses one vesus all.
            :param x: input instance.
            :return: a number as label
        """
        net_inputs = []

        # calculate net input for diffrent adalines
        for i in range(1, self.__number_of_class + 1):
            net_input = self.__adalines[i-1].net_input(x)
            net_inputs.append(net_input)

        label = np.argmax(net_inputs)+1
        return label


#####################################################
#                                                   #
#           test part for testing correctness of    #
#                  adaline_onw_vs_all class      #
#                                                   #
#####################################################
if __name__ == '__main__':

    # small example for testing adaline class
    import matplotlib.pyplot as plt
    from read_iris import read_iris_dataset
    from sklearn import preprocessing

    # read train and test from iris dataset
    trainX, validX, testX, trainY, validY, testY = read_iris_dataset(rand_state=0,
                                                                     train_ratio=0.60,
                                                                     valid_ratio=0.20,
                                                                     test_ratio=0.20)
    # print(trainX, testX, trainY, testY)

    # standard scaler
    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)
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
    units = adaline_one_vs_all(number_classes=3,learning_rate=0.01,max_epoch=200)
    units.fit(X=trainX, y=trainY, validX=validX, validY=validY, classes_name=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
              rand_state=0, plotting=True)

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
    print('Train Error= ', train_error / len(pred))

    # error of adalines(one vs all) for test set
    pred = []
    for x in testX:
        prediction = units.predict(x)
        pred.append(prediction)

    test_error = 0
    for _ in range(0, len(pred)):
        if pred[_] != testY[_]:
            test_error += 1
    print('Test Error= ', test_error/ len(pred))

    plt.show(block=True)
    print('======================')
