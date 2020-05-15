"""This file contains a class for adaline unit"""
import numpy as np
import matplotlib.pyplot as plt

class adaline:
    """
    Attributes:
        __learning_rate: learning rate of Perceptron
        __weights: weights of adaline
        __maximum_epoch: maximum number of epochs that adaline is
                        allowed to use for training
        __errors: number of misclassified instances in each epoch
        __weights_epochs: weights of adaline in each epoch
        __cut_error: stop training if mse error fall under specific value
    """

    def __init__(self, learning_rate=0.01, max_epoch=100, cut_error=None):
        """
        This function is constructor of adaline which sets
        different parameter of adaline.
        :param learning_rate: learning rate of adaline
        :param max_epoch: maximum number of iteration that adaline is
                        allowed to use for training
        :param: cut_error: stop training if mse error fall under specific value
        """
        # set learning rate
        self.__learning_rate = learning_rate

        # add weights attribute
        self.__weights = np.zeros(1)

        # set maximum number of epochs
        self.__max_epoch = max_epoch

        # error in different epochs
        self.__errors = []

        # rario of misclassified instances in each epoch for validation set
        self.__valid_errors = []

        # stop training if error fall under specific value, if it's none, don't consider it
        self.__cut_error = cut_error

    def net_input(self, X):
        """
        Net input of adaline for instance X, which equal: transpose(W)*X+W[0]
        :param X: Input instance
        :return: net input of adaline
        """
        #print('X: ', X)
        #print('W ', self.__weights)

        # net input for instance X
        net_input_x = np.dot(X, self.__weights[1::].T) + self.__weights[0]
        #print(net_input_x)

        return net_input_x

    #def activation(self, net_inp):
    #    """
    #    This function calculate activation of adaline(lin),
    #    :param net_inp: net input of adaline
    #                    np.array[net input1,...,net inputM]
    #    :return: activation of net input
    #    """
    #    # output of perceptron
    #    out = net_inp
    #    return out

    def predict(self, X):
        """
        This function predict output of adaline for instance X
        :param X: input instance(s).
        :return: corresponding activation function
        """
        net_inputs = self.net_input(X)
        #activation_out = self.activation(net_inp=net_inputs)
        activations_out = np.where(net_inputs >= 0.0, 1, -1)

        return activations_out

    def fit(self, X, y, validX=None, validY=None, class_name="", rand_state=0, plotting=False):
        """
        This method of adaline class gets training dataset and
        its label, then it trains adaline. after doing "__max_epoch" epochs,
        it stops training, or when it fall under cut off error.
        :param X: Training set features: numpy array of numpy arrays
                    [[feature1 x1, feature2 x2,...,featuren xn],
                    ...,
                    [feature1 xm, feature2 xm,...,featuren xm]]
        :param y: Training set labels: numpy array
                    [[label x1], [label x2], ..., [label xn]]
        :param validX: validation set (optional)
        :param validY: labels of validation set (optional)
        :param rand_state: seed for random number
        :param plotting: if it's true, this function plots
                        cost and weights during learning
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
            plt.title(class_name+': Training MSE Error during training')
            plt.xlabel('Epochs')
            plt.ylabel('Erorr')
            plt.ion()

            if validX is not None and validY is not None:
                fig_valid_errors = plt.figure()
                plt.title(class_name + ': Validation Error during training')
                plt.xlabel('Epochs')
                plt.ylabel('Erorr')
                plt.ion()

            fig_weights = plt.figure()
            plt.title(class_name+': Weights during training')
            plt.xlabel('Epochs')
            plt.ylabel('Weights')
            plt.ion()

            # colors for plotting
            colors = [plt.cm.jet(i) for i in np.linspace(0, 1, X.shape[1] + 2)]

        # for maximum number of epochs train adaline by training set
        for epoch in range(0, self.__max_epoch):

            # fit adaline by giving all instances of training set
            # one by one
            mse_error = 0
            for x, target in zip(X, y):

                # output of adaline for sample x(before thresholding)
                y_hat = self.net_input(x)
                #print(target[0])
                #print(y_hat)
                # update weights
                self.__weights[1::] += self.__learning_rate * (target[0] - y_hat) * x
                self.__weights[0] += self.__learning_rate * (target[0] - y_hat)


                # count misclassified instances
                mse_error += (target - y_hat) ** 2

            #print(y_hat)
            # error of epoch
            mse_error = mse_error / X.shape[0]

            # keep error of each epoch
            self.__errors.append(mse_error)

            # calculate validation set error
            error_valid = 0
            if (validX is not None) and (validY is not None):
                for x_valid, target_valid in zip(validX, validY):
                    # output of adaline for sample x
                    y_hat = self.predict(x_valid)
                    # count misclassified instances
                    if (target_valid - y_hat) != 0.0:
                        error_valid += 1
                # error of validation of epoch
                error_valid = error_valid / validX.shape[0]
                self.__valid_errors.append(error_valid)

            # keep weights of each epochs
            self.__weights_epochs = np.hstack((self.__weights_epochs, np.matrix(self.__weights).T))

            if (plotting == True) and (epoch % 10 == 0 or epoch==self.__max_epoch or (self.__cut_error!=None and mse_error<self.__cut_error)):
                # plot error after each epoch
                plt.figure(fig_errors.number)
                plt.plot(range(1, len(self.__errors)+1), self.__errors, color=colors[0],  marker='o', markersize=3)
                #plt.pause(0.05)

                if (validX is not None) and (validY is not None):
                    # plot error of validation after each epoch
                    plt.figure(fig_valid_errors.number)
                    plt.plot(range(1, len(self.__valid_errors) + 1), self.__valid_errors, color=colors[1])

                # plot weights
                plt.figure(fig_weights.number)
                plt.legend(loc=3)
                for idx, w in enumerate(self.__weights):
                    plt.pause(0.001)
                    plt.plot(range(1,self.__weights_epochs[idx].shape[1]+1), self.__weights_epochs[idx][0].tolist()[0],
                             color=colors[idx], linestyle='-', marker='o', markersize=2,
                             label='w{0}'.format(idx) if epoch==0 else '')


            # if error of classifying is lower than self.__cut_error , stop doing
            # more epochs
            if self.__cut_error != None and mse_error < self.__cut_error:
                break

        return self

    def load_weights(self, weights):
        """
        Load weights of adaline from file
        :param weights: a list in form [w0, w1, w2, ..., wn]
        :return:
        """
        self.__weights = np.array(weights)
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
#                  adaline    class                 #
#                                                   #
#####################################################
if __name__ == '__main__':

    # small example for testing adaline class
    from sklearn import preprocessing
    import matplotlib.pyplot as plt
    from read_iris import read_iris_dataset

    # read train and test from iris dataset
    trainX, validX, testX, trainY, validY, testY = read_iris_dataset(rand_state=0,
                                                                     train_ratio=0.7,
                                                                     valid_ratio=0.15,
                                                                     test_ratio=0.15)

    # standard scaler
    scaler = preprocessing.StandardScaler().fit(trainX)
    trainX = scaler.transform(trainX)
    validX = scaler.transform(validX)
    testX = scaler.transform(testX)

    #print(trainX, testX, trainY, testY)

    # for classifying Iris setosa
    #trainY = np.where(trainY == 'Iris-setosa', 1, -1)
    #testY = np.where(testY == 'Iris-setosa', 1, -1)

    # for classifying Iris-versicolor
    trainY = np.where(trainY == 'Iris-versicolor', 1.0, -1.0)
    validY = np.where(validY == 'Iris-versicolor', 1.0, -1.0)
    testY = np.where(testY == 'Iris-versicolor', 1.0, -1.0)

    # for classifying Iris-versicolor
    #trainY = np.where(trainY == 'Iris-virginica', 1, -1)
    #testY = np.where(testY == 'Iris-virginica', 1, -1)

    unit = adaline(learning_rate=0.01, max_epoch=200)
    unit.fit(X=trainX, y=trainY,  validX=validX, validY=validY,class_name='Iris-versicolor', plotting=True, rand_state=0)


    print('learning rate: ', unit.get_learning_rate())
    print('weights: ', unit.get_weights())

    train_error = unit.get_error_in_epochs()
    train_error = train_error[len(train_error) - 1]
    print('Train MSE-Error= ', train_error)

    validation_error = unit.get_valid_error_in_epochs()
    if len(validation_error) > 0:
        validation_error = validation_error[len(validation_error) - 1]
        print('Validation MSE-Error= ', validation_error)

    pred = unit.predict(testX)
    test_error = 0
    for _ in range(0, pred.shape[0]):
        if pred[_] != testY[_]:
            test_error += 1
    print('Test Error= ', test_error / pred.shape[0])
    plt.show(block=True)
