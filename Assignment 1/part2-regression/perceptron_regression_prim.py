"""This file contains a class for regression perceptron unit"""
import numpy as np
import matplotlib.pyplot as plt
import json

class regression_perceptron:
    """
    Attributes:
        __learning_rate: learning rate of regression Perceptron
        __weights: weights of regression perceptron
        __maximum_epoch: maximum number of epochs that regression perceptron is
                        allowed to use for training
        __errors: ratio of misclassified instances in each epoch
        __valid_errors: rario of misclassified instances in each epoch for validation set
        __weights_epochs: weights of regression perceptron in each epoch
        __cut_error: stop training if error fall under specific value
    """

    def __init__(self, learning_rate=0.01, max_epoch=100, cut_error=None):
        """
        This function is constructor of regression Perceptron which sets
        different parameter of regression Perceptron.
        :param number_of_classes: number of different labels
        :param learning_rate: learning rate of regression Perceptron for updating weights
        :param max_epoch: maximum number of iterations that regression perceptron is
                        allowed to use for training
        :param: cut_error: stop training if error fall under specific value
        """
        # set learning rate
        self.__learning_rate = learning_rate

        # add weights attribute
        self.__weights = np.zeros(1)

        # set maximum number of epochs
        self.__max_epoch = max_epoch

        # error in different epoch
        self.__errors = []

        # rario of misclassified instances in each epoch for validation set
        self.__valid_errors = []

        # stop training if error fall under specific value, if it's none don't consider it
        self.__cut_error = cut_error

    def net_input(self, X):
        """
        Net input of regression perceptron for instance X, which equal: transpose(W)*X+W[0]
        :param X: Input instance
        :return: net input of regression perceptron
        """
        #print('X: ', X)
        #print('W ', self.__weights)

        # net input for instance X
        net_input_x = np.dot(X, self.__weights[1::].T) + self.__weights[0]
        #print(net_input_x)

        return net_input_x

    def activation(self, net_inp):
        """
        This function calculate activation of percepton
        :param net_inp: net input of regression perceptron
                        np.array[net input1,...,net inputM]
        :return: activation of net input
        """
        out = net_inp
        return out

    def predict(self, X):
        """
        This function predict output of regression perceptron for instance(s) X
        :param X: input instance(s).
                  format: np.array[instance1 np.array,...,instance1 np.arrayM]
        :return: corresponding activation function
        """
        net_inputs = self.net_input(X)
        activations_out = self.activation(net_inp=net_inputs)
        return activations_out

    def fit(self, X, y, validX=None, validY=None, class_name="", rand_state=0, plotting=False, write_in_file=False, feature_i=None):
        """
        This method of regression perceptron class gets training dataset and
        its label, then it trains regression perceptron. after doing "__max_epoch" epochs,
        it will stop training.
        :param X: Training set features: numpy array of numpy arrays
                    [[feature1 x1, feature2 x2,...,featuren xn],
                    ...,
                    [feature1 xm, feature2 xm,...,featuren xm]]
        :param y: Training set labels: numpy array
                    [[label x1], [label x2], ..., [label xn]]
        :param validX: validation set (optional)
        :param validY: labels of validation set (optional)
        :param rand_state: seed for generating random number
        :param plotting: if it's true, this function plots
                        cost and weights during learning
        :param feature_i: if not None and equal integer i plot y according to feature i and draw prediction line
        :return: self
        """

        # initial weights: (number of features + 1) weights are needed
        #self.__weights = np.zeros(X.shape[1] + 1)
        np.random.seed(rand_state)
        self.__weights = np.random.normal(0, 1, X.shape[1] + 1)

        self.__errors = []
        self.__valid_errors = []

        # keep weights of each epochs
        self.__weights_epochs = np.matrix(np.matrix(self.__weights).T)

        if plotting == True:

            fig_errors = plt.figure()
            plt.title(class_name+': Train MSE Errors during training')
            plt.xlabel('Epochs')
            plt.ylabel('Erorr')
            plt.ion()

            if validX is not None and validY is not None:
                fig_valid_errors = plt.figure()
                plt.title(class_name + ': Validation MSE Errors during training')
                plt.xlabel('Epochs')
                plt.ylabel('Erorr')
                plt.ion()

            fig_weights = plt.figure()
            plt.title(class_name+': Weights during training')
            plt.xlabel('Epochs')
            plt.ylabel('Weights')
            plt.ion()

            if feature_i is not None:
                fig_mpg_weight = plt.figure()
                plt.title(class_name + ': MPG-Weight')
                plt.xlabel('Weight (scaled)')
                plt.ylabel('MPG')
                plt.ion()

            # colors for plotting
            colors = [plt.cm.jet(i) for i in np.linspace(0, 1, X.shape[1] + 2)]

        # for maximum number of epochs train regression perceptron by training set
        for epoch in range(0, self.__max_epoch):

            # fit regression perceptron by giving all instances of training set
            # one by one
            error = 0
            for x, target in zip(X, y):

                # output of regression perceptron for sample x
                y_hat = self.predict(x)

                # update weights
                self.__weights[1::] += self.__learning_rate * (target - y_hat) * x
                self.__weights[0] += self.__learning_rate * (target - y_hat)


                # calculate MSE error
                error += (target - y_hat) ** 2
            # error of epoch
            error = error / X.shape[0]

            # keep error of each epoch
            self.__errors.append(error)

            # calculate validation set MSE error
            error_valid = 0
            if (validX is not None) and (validY is not None):
                for x_valid, target_valid in zip(validX, validY):
                    # output of regression perceptron for sample x
                    y_hat = self.predict(x_valid)
                    # amount of miss prediction
                    error_valid += (target_valid - y_hat)**2
                # error of validation of epoch
                error_valid = error_valid / validX.shape[0]
                self.__valid_errors.append(error_valid)

            # keep weights of each epochs
            self.__weights_epochs = np.hstack((self.__weights_epochs, np.matrix(self.__weights).T))

            if plotting == True and (epoch % 10 == 0 or epoch==self.__max_epoch or (self.__cut_error!=None and error<self.__cut_error)):
                # plot MSE error after each epoch
                plt.figure(fig_errors.number)
                plt.plot(range(1, len(self.__errors)+1), self.__errors, color=colors[0],  marker='o', markersize=3)
                #plt.pause(0.001)

                if (validX is not None) and (validY is not None):
                    # plot error of validation after each epoch
                    plt.figure(fig_valid_errors.number)
                    plt.plot(range(1, len(self.__valid_errors) + 1), self.__valid_errors, color=colors[1])
                    #plt.pause(0.001)

                # plot weights
                plt.figure(fig_weights.number)
                plt.legend(loc=3)
                for idx, w in enumerate(self.__weights):
                    plt.plot(range(1,self.__weights_epochs[idx].shape[1]+1), self.__weights_epochs[idx][0].tolist()[0],
                             color=colors[idx], linestyle='-', marker='o', markersize=2,
                             label='w{0}'.format(idx) if epoch==0 else '')
                    plt.pause(0.001)

                if feature_i is not None:
                    # plot mpg-weight
                    plt.figure(fig_mpg_weight.number)
                    #plt.legend(loc=3)
                    plt.clf()
                    plt.scatter(x=X[:,feature_i], y=y, c='red')
                    weight_x = np.linspace(start=-2, stop=2, num=100)
                    _x = np.zeros((100, self.__weights.shape[0]-1))
                    _x[:,feature_i] = weight_x
                    _x[:, 9] = weight_x ** 2
                    plt.plot(weight_x, [self.predict(xi) for xi in _x],'b--')
                    plt.pause(0.001)

            # if error of regression is lower than self.__cut_error , stop doing
            # more epochs
            if self.__cut_error != None and error < self.__cut_error:
                break

        # write weights in file
        if write_in_file == True:
            with open('output-regression-peceptron.json', 'w') as outputfile:
                json.dump(self.__weights.tolist(), outputfile)

        return self

    def load_weights(self, weights):
        """
        Load weights of regression perceptron from file
        :param weights: a list in form [w0, w1, w2, ..., wn]
        :return:
        """
        self.__weights = np.array(weights)
        return self

    def load_regression_perceptron_from_file(self, file_name):
        '''
        This function read weights of trained regression perceptron from file
        :param file_name: it's name of a json file with format like: 'input.json'
                with code should be in same directory
        :return:
        '''
        # read weights from file
        with open(file_name) as json_file:
            weights = json.load(json_file)
        self.load_weights(weights=weights)
        return self

    def get_learning_rate(self):
        """Return Learning Rate"""
        return self.__learning_rate

    def get_weights(self):
        """Return Weights"""
        return self.__weights

    def get_valid_error_in_epochs(self):
        """Return error in each epochs"""
        return self.__valid_errors

    def get_error_in_epochs(self):
        """Return error in each epochs"""
        return self.__errors

#####################################################
#                                                   #
#           test part for testing correctness of    #
#                  regression Perceptron class      #
#                                                   #
#####################################################
if __name__ == '__main__':

    # small example for testing regression perceptron class

    import matplotlib.pyplot as plt
    from read_auto_mpg_dataset import read_auto_mpg_dataset_continuous, read_auto_mpg_dataset
    from sklearn import preprocessing

    # read train and test from iris dataset
    #trainX, validX, testX, trainY, validY, testY = read_iris_dataset(rand_state=0,
    #                                                 train_ratio=0.60,
    #                                                 valid_ratio=0.20,
    #                                                 test_ratio=0.20) # epoch 200 rate:0.01

    # read train and test from iris dataset
    #trainX, validX, testX, trainY, validY, testY = read_auto_mpg_dataset_continuous(rand_state=5,
    #                                                                 train_ratio=0.70,
    #                                                                 valid_ratio=0.15,
    #                                                                 test_ratio=0.15, order='2')
    #i_feature=2
    trainX, validX, testX, trainY, validY, testY = read_auto_mpg_dataset(rand_state=5,
                                                                        train_ratio=0.70,
                                                                        valid_ratio=0.15,
                                                                        test_ratio=0.15, order='2')
    i_feature = 3

    # standard scaler
    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)
    validX = scaler.transform(validX)
    testX = scaler.transform(testX)

    #print(trainX.shape[0], validX.shape[0], testX.shape[0], trainY.shape[0], validY.shape[0], testY.shape[0])

    # create and train regression perceptron
    unit = regression_perceptron(learning_rate=0.002, max_epoch=500)
    unit.fit(X=trainX, y=trainY, validX=validX, validY=validY, plotting=True, feature_i=i_feature)

    print('learning rate: ', unit.get_learning_rate())
    print('weights: ',unit.get_weights())

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

    #print(testY)
    #print(pred)
    #for x,y in zip(testY, pred):
    #    print(x,y)

    test_error = 0
    for _ in range(0, pred.shape[0]):
        if pred[_] != testY[_]:
            test_error += (pred[_] - testY[_])**2
    print('Test MSE Error= ', test_error/pred.shape[0])
    plt.show(block=True)

