"""
Multi-Layer Perceptron
"""
import numpy as np
from activation_functions import activation, activation_derivative
import copy
import json


class MLP:

    def __init__(self, n, activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=300, mode='batch', random_state=None, plot_info=False, cut_cost=None, save=True):
        """
        Constructor of class MLP
        :param n: is a list which n[i] demonstrates number of neurons
                    in layer i, n[0] is number of features
        :param activations: is a list which activations[i] is a string
                that is activation of layer i
        :param type_of_cost: a string 'MSE' or 'cross_entropy'
        :param learning_rate: learning rate for updating learnable parameters
        :param max_epoch: maximum epoch that the network is allowed to do
        :param mode: is a string that is equal 'batch' or 'stochastic' or 'momentum' or 'steepest descent'
        :param random_state: random state for generating weights and biases and making algorithms
                            reproducible for different runs.
        :param plot_info: if it be true, MLP class saves all weights and biases for each epoch
        :param cut_cost: if it be none, it will be ignored, otherwise during training, if cost fall under
                        cost_cut, training is finished.
        """

        # when we want load mlp from file we don't care these values
        # otherwise validity of them should be checked
        if len(n) != 0 and len(activations) != 0:
            if len(n)-1 != len(activations):
                raise ValueError('Given number of layers and number of '+
                                 'activation for each layer aren\'t equal!')

        # type of cost function
        self.__type_cost = type_of_cost

        # is a list which __n[i] demonstrates number of neurons in layer i, __n[0] is number of features
        self.__n = np.copy(n).tolist()

        # is a list which activations[i] demonstrates is a string that is activation of layer i
        self.__activations = np.copy(activations).tolist()

        # learning rate for updating laudable parameters
        self.__learning_rate = learning_rate

        # maximum epoch that the network is allowed to do
        self.__max_epoch = max_epoch

        # if true save network after training
        self.__save = save

        # is a string that is equal 'batch' or 'stochastic' or 'momentum' or 'steepest'
        self.__mode = mode
        if self.__mode != 'batch' and self.__mode != 'stochastic' and self.__mode != 'momentum' and self.__mode != 'steepest':
            raise ValueError("Mode should be batch or stochastic or momentum or steepest!")

        # element i in self.__weights is a n*m ndarray which is weights matrix of
        # layer i
        # Wij[l]: weight of neuron j in layer l-1 to i th neuron in layer l
        self.__weights = []

        # it determines save more extra information for plotting or not
        self.plot_info = plot_info

        if self.plot_info == True:
            self.weigths_epochs = []
            self.biases_epochs = []
            self.cost_epochs = []
            self.dI_epochs = None
            if self.__mode == 'momentum':
                self.dI_epochs = []

        self.__cut_cost = cut_cost

        # determined MLP trained or not
        self.trained = False

        # Momentum for weights
        self.__Vweights = []

        # element i in self.__bias is a n*1 ndarray which is bias matrix of
        # layer i
        self.__bias = []

        # Momentum for bias
        self.__Vbias = []

        # number of layers
        self.__num_layers = len(n)

        # seed random number
        if random_state is not None:
            np.random.seed(random_state)

        for l in range(1, self.__num_layers):

            # initial weights of different layers
            self.__weights.append(np.random.randn(self.__n[l], self.__n[l-1]) * np.sqrt(1/ self.__n[l-1]))

            # initial bias matrix
            self.__bias.append(np.zeros([self.__n[l], 1], dtype=np.float64))

            # set parameters for updating with momentum
            self.__Vweights.append(np.zeros([self.__n[l], self.__n[l - 1]], dtype=np.float64))
            self.__Vbias.append(np.zeros([self.__n[l], 1], dtype=np.float64))

        # coefficient of weights and bias if momentum used
        self.__beta_momentum = 0.9

    def cost_function(self, target, y):
        """
        Cost function of MLP can be MSE and cross_entropy
        :param target: actual labels (correct labels)
        :param y: output labels of MLP
                    [instance1 LABEL, instance2  LABEL, ..., instanceN  LABEL]
        """
        # number of samples
        num_sample = y.shape[1]

        cost = 0
        if self.__type_cost == 'MSE':
            # MSE Cost Function
            cost = np.sum((y-target)**2) / (2*num_sample)
        elif self.__type_cost == 'cross_entropy':
            # Cross Entropy

            # if cost function is cross-entropy, normalize output layer by some of outputs
            # sum of outputs of output layer for each sample
            sum_outputs = np.sum(y, axis=0)
            inverse_sum_outputs = 1 / sum_outputs
            norm_y = np.multiply(y, inverse_sum_outputs)

            cost = -1 * np.sum( np.multiply(target, np.log(norm_y)) +
                                np.multiply((1-target), np.log(1-norm_y))) / (num_sample)

            #cost = -1 * np.sum( np.multiply(target, np.log(y)) +
            #                    np.multiply((1-target), np.log(1-y))) / (num_sample)
        else:
            raise ValueError('Cost function type should be MSE or cross_entropy!')
        return cost

    def forward_propagation(self, X):
        """
        Do forward propagation
        :param X: is an numpy ndarray of instance(s) in form:
                    [instance1, instance2, ..., instanceN]
        :return:    a list that contains net input(I) of each layer
                    a list that contains output(Y) of each layer
        """
        # catch for net input(I) of each layer during forward propagation
        catch_I = [X]
        # catch for output(Y) of each layer during forward propagation
        catch_Y = [X]

        # forward propagation
        for l in range(0, self.__num_layers-1):
            # calculate net input of layer l
            net_input_l = np.dot(self.__weights[l], catch_Y[l]) + self.__bias[l]

            # calculate output of layer l
            output_l = activation[self.__activations[l]](net_input_l)

            # add net input and output of layer l to catch
            catch_I.append(net_input_l)
            catch_Y.append(output_l)

        # output net input and output of each layers
        return catch_I, catch_Y

    def backward_propagation(self, target, catch_I, catch_Y):
        """
        This function does backward propagation and calculated gradients
        :param catch_I: catch for net input(I) of each layer during forward propagation
        :param catch_Y: catch for output(Y) of each layer during forward propagation
        :param target: true labels of instance(s)
        :return
        """
        # start time
        #start_time = time.time()

        # number of samples
        num_sample = target.shape[1]

        # grad of w
        grad_w = []
        for w in self.__weights:
            grad_w.append(np.zeros(w.shape))

        # grad of b
        grad_b = []
        for b in self.__bias:
            grad_b.append(np.zeros(b.shape))

        #  d(cost)/ d(net_input)
        d_I = [None] * len(self.__weights)

        # calculate error and grad of output layer
        l = len(self.__weights) - 1
        if self.__type_cost == 'MSE':
            # d(cost)/ d(net_input) = -(t-y).f-derivative(net input)
            d_I[l] = -1 * np.multiply((target-catch_Y[l+1]),
                                                   activation_derivative[self.__activations[l]](catch_I[l+1]))

            # d(cost)/d(w) = (1/number of sample) * (d_I[l] . transpose(y)[l-1])
            grad_w[l] = (1/num_sample) * np.dot(d_I[l], catch_Y[l].T)
            grad_b[l] = (1/num_sample) * np.sum(d_I[l], axis=1, keepdims=True)

        elif self.__type_cost == 'cross_entropy':
            # d(cost)/ d(net_input) = (-(y/t) + (1-y)/(1-t)).f-derivative(net input)
            #d_I[l] = np.multiply((-1*(np.divide(target, catch_Y[l+1]))+
            #                          (np.divide(1-target, 1-catch_Y[l+1]))),
            #                                       activation_derivative[self.__activations[l]](catch_I[l+1]))

            # if cost function is cross-entropy, normalize output layer by some of outputs
            # sum of outputs of output layer for each sample
            sum_outputs = np.sum(catch_Y[l + 1], axis=0)
            inverse_sum_outputs = 1 / sum_outputs
            norm_y = np.multiply(catch_Y[l + 1], inverse_sum_outputs)

            d_I[l] = np.multiply((-1 * (np.divide(target, norm_y)) +
                                  (np.divide(1 - target, 1 - norm_y))),
                                 np.multiply(activation_derivative[self.__activations[l]](catch_I[l + 1]), inverse_sum_outputs))

            # d(cost)/d(w) =
            grad_w[l] = (1 / num_sample) * np.dot(d_I[l], catch_Y[l].T)
            grad_b[l] = (1 / num_sample) * np.sum(d_I[l], axis=1, keepdims=True)

            # d(cost)/d(w) =
            #grad_w[l] = (1 / num_sample) * np.dot(d_I[l], catch_Y[l].T)
            #grad_b[l] = (1 / num_sample) * np.sum(d_I[l], axis=1, keepdims=True)


        else:
            raise ValueError('Type of cost function isn\t correct!')

        # grad of hidden layer
        for l in range(len(self.__weights)-2, -1, -1):
            d_I[l] = np.multiply(np.dot(self.__weights[l+1].T, d_I[l+1]), (activation_derivative[self.__activations[l]](catch_I[l+1])))
            # - d(cost)/d(w) = (1/number of sample) * (d_I[l] . transpose(y)[l-1])
            grad_w[l] = (1 / num_sample) * np.dot(d_I[l], catch_Y[l].T)
            grad_b[l] = (1 / num_sample) * np.sum(d_I[l], axis=1, keepdims=True)

        # end time
        #end_time = time.time()
        #print('batch time: {}'.format(end_time - start_time))

        return grad_w, grad_b

    def backward_propagation_stochastic(self, target, catch_I, catch_Y):
        """
        Do backward propagation
        :param catch_I: catch for net input(I) of each layer during forward propagation
        :param catch_Y: catch for output(Y) of each layer during forward propagation
        :param target: true labels of instance(s)
        :return
        """

        # start time
        #start_time = time.time()

        # grad of w
        grad_w = []
        for w in self.__weights:
            grad_w.append(np.zeros(w.shape))

        # grad of b
        grad_b = []
        for b in self.__bias:
            grad_b.append(np.zeros(b.shape))

        #  d(cost)/ d(net_input)
        d_I = [None] * len(self.__weights)

        # calculate error and grad of output layer
        l = len(self.__weights) - 1
        if self.__type_cost == 'MSE':
            # d(cost)/ d(net_input) = -(t-y).f-derivative(net input)
            d_I[l] = -1 * np.multiply((target-catch_Y[l+1]),
                                                   activation_derivative[self.__activations[l]](catch_I[l+1]))

            # d(cost)/d(w) = (1/number of sample) * (d_I[l] . transpose(y)[l-1])
            grad_w[l] = np.dot(d_I[l], catch_Y[l].T)
            grad_b[l] = d_I[l]

        elif self.__type_cost == 'cross_entropy':
            # d(cost)/ d(net_input) = (-(y/t) + (1-y)/(1-t)).f-derivative(net input)
            d_I[l] = np.multiply((-1 * (np.divide(target, catch_Y[l + 1])) +
                                  (np.divide(1 - target, 1 - catch_Y[l + 1]))),
                                 activation_derivative[self.__activations[l]](catch_I[l + 1]))

            # d(cost)/d(w) =
            grad_w[l] = np.dot(d_I[l], catch_Y[l].T)
            grad_b[l] = np.sum(d_I[l], axis=1, keepdims=True)
        else:
            raise ValueError('Type of cost function isn\t correct!')

        # grad of hidden layer
        for l in range(len(self.__weights)-2, -1, -1):
            d_I[l] = np.multiply(np.dot(self.__weights[l+1].T, d_I[l+1]), (activation_derivative[self.__activations[l]](catch_I[l+1])))
            # - d(cost)/d(w) = (1/number of sample) * (d_I[l] . transpose(y)[l-1])
            grad_w[l] = np.dot(d_I[l], catch_Y[l].T)
            grad_b[l] = d_I[l]

        # end time
        #end_time = time.time()
        #print('Stocastic time: {}'.format(end_time-start_time))

        return grad_w, grad_b

    def fit_batch(self, X, y):
        """
        Learn learnable parameters by using back propagation in batch mode
        :param X: training dataset, each column is an observation
        :param y: label of training dataset
        :return:
        """

        # calculate cost at beginning
        catch_I, catch_Y = self.forward_propagation(X=X)
        cost = self.cost_function(target=y, y=catch_Y[len(catch_Y) - 1])

        if self.plot_info == True:
            self.cost_epochs.append(cost)
            self.weigths_epochs.append(copy.deepcopy(self.__weights))
            self.biases_epochs.append(copy.deepcopy(self.__bias))
            print('Iter {}/{}, Cost {} ...'.format(0, self.__max_epoch, cost))

        # do 'self.__max_epoch' iteration for coverage and reducing cost
        for epoch in range(0, self.__max_epoch):

            if self.__mode == 'batch':
                # give whole dataset to network

                # do forward propagation
                catch_I, catch_Y = self.forward_propagation(X=X)


                # calculate gradients by back propagation
                grad_w, grad_b = self.backward_propagation(target=y, catch_I=catch_I, catch_Y=catch_Y)

                # update weights
                self.update_weight_bias_normal(grad_w=grad_w, grad_b=grad_b)

                # calculate cost
                cost = self.cost_function(target=y, y=catch_Y[len(catch_Y)-1])

                if self.plot_info == True:
                    self.cost_epochs.append(cost)
                    self.weigths_epochs.append(copy.deepcopy(self.__weights))
                    self.biases_epochs.append(copy.deepcopy(self.__bias))
                    print('Iter {}/{}, Cost {} ...'.format(epoch+1, self.__max_epoch, cost))

                # if self.__cost_cut isn't None and cost is lower than it stop training
                if self.__cut_cost is not None:
                    if cost < self.__cut_cost:
                        break

    def fit_momentum_stocastic(self, X, y):
        """
        Learn learnable parameters by using back propagation with momentum
        :param X: training dataset, each column is an observation
        :param y: label of training dataset
        :return:
        """

        # calculate cost at beginning
        catch_I, catch_Y = self.forward_propagation(X=X)
        cost = self.cost_function(target=y, y=catch_Y[len(catch_Y) - 1])

        if self.plot_info == True:
            self.cost_epochs.append(cost)
            self.weigths_epochs.append(copy.deepcopy(self.__weights))
            self.biases_epochs.append(copy.deepcopy(self.__bias))
            if self.dI_epochs is not None:
                self.dI_epochs.append(copy.deepcopy(self.__bias))
            print('Iter {}/{}, Cost {} ...'.format(0, self.__max_epoch, cost))

        # do 'self.__max_epoch' iteration for coverage and reducing cost
        for epoch in range(0, self.__max_epoch):

            # give observations in dataset to newwork one by one
            for index in range(0, X.shape[1]):

                # start_time = time.time()

                # do forward propagation
                catch_I, catch_Y = self.forward_propagation(X=X[:, index:(index + 1)])

                # calculate gradients by back propagation
                grad_w, grad_b = self.backward_propagation_stochastic(target=y[:, index:(index + 1)],
                                                                          catch_I=catch_I, catch_Y=catch_Y)
                # update weights
                if self.__mode == 'stochastic':
                    # if mode stochastic use normal manner for updating weights
                    self.update_weight_bias_normal(grad_w=grad_w, grad_b=grad_b)
                elif self.__mode == 'momentum':
                    # if mode momentum use momentum manner for updating weights
                    self.update_weight_bias_momentum(grad_w=grad_w, grad_b=grad_b)

                    # end_time = time.time()
                    # print('{} time: {}'.format(self.__mode, end_time - start_time))

                # calculate cost at the end of epoch
            catch_I, catch_Y = self.forward_propagation(X=X)
            cost = self.cost_function(target=y, y=catch_Y[len(catch_Y) - 1])

            if self.plot_info == True:
                self.cost_epochs.append(cost)
                self.weigths_epochs.append(copy.deepcopy(self.__weights))
                self.biases_epochs.append(copy.deepcopy(self.__bias))
                if self.dI_epochs is not None:
                   self.dI_epochs.append(copy.deepcopy(grad_b))
                print('Iter {}/{}, Cost {} ...'.format(epoch + 1, self.__max_epoch, cost))

            # if self.__cost_cut isn't None and cost is lower than it stop training
            if self.__cut_cost is not None:
                if cost < self.__cut_cost:
                    break

    def fit_steepest_descent(self, X, y):
        """
        Learn learnable parameters by using back propagation with steepest descent
        :param X: training dataset, each column is an observation
        :param y: label of training dataset
        :return:
        """

        # calculate cost at beginning
        catch_I, catch_Y = self.forward_propagation(X=X)
        cost = self.cost_function(target=y, y=catch_Y[len(catch_Y) - 1])

        if self.plot_info == True:
            self.cost_epochs.append(cost)
            self.weigths_epochs.append(copy.deepcopy(self.__weights))
            self.biases_epochs.append(copy.deepcopy(self.__bias))
            print('Iter {}/{}, Cost {} ...'.format(0, self.__max_epoch, cost))

        # do 'self.__max_epoch' iteration for coverage and reducing cost
        for epoch in range(0, self.__max_epoch):

            # calculate cost at beginning
            catch_I, catch_Y = self.forward_propagation(X=X)
            pre_cost = self.cost_function(target=y, y=catch_Y[len(catch_Y) - 1])

            # keep a copy of weights and bias
            copy_weights = copy.deepcopy(self.__weights)
            copy_bias = copy.deepcopy(self.__bias)

            imporvement = False
            while not imporvement:
                # give observations in dataset to newwork one by one
                for index in range(0, X.shape[1]):

                    # start_time = time.time()

                    # do forward propagation
                    catch_I, catch_Y = self.forward_propagation(X=X[:, index:(index + 1)])

                    # calculate gradients by back propagation
                    grad_w, grad_b = self.backward_propagation_stochastic(target=y[:, index:(index + 1)],
                                                                              catch_I=catch_I, catch_Y=catch_Y)
                    # update weights
                    # if mode stochastic use normal manner for updating weights
                    self.update_weight_bias_normal(grad_w=grad_w, grad_b=grad_b)


                    # end_time = time.time()
                    # print('{} time: {}'.format(self.__mode, end_time - start_time))

                    # calculate cost at the end of epoch
                catch_I, catch_Y = self.forward_propagation(X=X)
                cost = self.cost_function(target=y, y=catch_Y[len(catch_Y) - 1])

                #print('Cost: ', cost)
                #print('preCost: ', pre_cost)
                if cost < pre_cost:
                    #print('Improvement occurred')

                    # double learning rate
                    self.__learning_rate = self.__learning_rate * 2
                    imporvement = True
                else:
                    #print('Improvement wasn\'t occurred')

                    # restore weights and biases and divide learning rate by 2
                    self.__learning_rate = self.__learning_rate / 2
                    self.__weights = copy.deepcopy(copy_weights)
                    self.__bias = copy.deepcopy(copy_bias)

            if self.plot_info == True:
                self.cost_epochs.append(cost)
                self.weigths_epochs.append(copy.deepcopy(self.__weights))
                self.biases_epochs.append(copy.deepcopy(self.__bias))
                print('Iter {}/{}, Cost {} ...'.format(epoch + 1, self.__max_epoch, cost))

            # if self.__cost_cut isn't None and cost is lower than it stop training
            if self.__cut_cost is not None:
                if cost < self.__cut_cost:
                    break

    def fit(self, X, y):
        """
        Learn learnabble parameters by using back propagation
        :param X: training dataset, each column is an observation
        :param y: label of training dataset
        :return:
        """
        if self.__mode == 'momentum' or self.__mode == 'stochastic':
            self.fit_momentum_stocastic(X=X, y=y)
        elif self.__mode == 'batch':
            self.fit_batch(X=X, y=y)
        elif self.__mode == 'steepest':
            self.fit_steepest_descent(X=X, y=y)
        else:
            raise ValueError('Mode is not correct!')

        # Set status of training of MLP as True
        self.trained = True

        if self.__save == True:
            # write MLP in file
            self.save()

    def update_weight_bias_normal(self, grad_w, grad_b):
        """
        This function gets gradient of w and b and updates them
        :param grad_w:
        :param grad_b:
        :return:
        """
        # Update weights and bias of each layer
        for l in range(0, len(self.__weights)):
            self.__weights[l] = self.__weights[l] - self.__learning_rate * grad_w[l]
            self.__bias[l] = self.__bias[l] - self.__learning_rate * grad_b[l]

    def update_weight_bias_momentum(self, grad_w, grad_b):
        """
        This function gets gradient of w and b and updates them.
        It considers momentum for updating.
        :param grad_w:
        :param grad_b:
        :return:
        """
        # Update weights and bias of each layer by using momentum term.
        for l in range(0, len(self.__weights)):

            # Update momentum
            self.__Vweights[l] = self.__beta_momentum * self.__Vweights[l] + (1-self.__beta_momentum) * grad_w[l]
            self.__Vbias[l] = self.__beta_momentum * self.__Vbias[l] + (1-self.__beta_momentum) * grad_b[l]

            # Update weights and bias with respect to momentum
            self.__weights[l] = self.__weights[l] - self.__learning_rate * self.__Vweights[l]
            self.__bias[l] = self.__bias[l] - self.__learning_rate * self.__Vbias[l]

    def predict(self, X):
        """
        Predict label of input instance(s)
        :param X:
        :return: predicted labels
        """

        # do propagation and find output of network for instance(S)
        catch_I, catch_Y = self.forward_propagation(X=X)
        output_layer = catch_Y[len(catch_Y)-1]

        # keep greatest output in output layer and set it one,
        # set others zero
        max_label = np.argmax(output_layer, axis=0)

        max_output = np.zeros(output_layer.shape)
        for idx_sample in range(0, output_layer.shape[1]):
            max_output[max_label[idx_sample], idx_sample] = 1

        return max_output

    def save(self):
        """
        save MLP into a file for future use, MLP is saved
        in MLP-output.json
        :return:
        """
        # create a dictionary of necessary info for saving in file
        mlp_dic = {'n': None, 'activations': None, 'type_cost': None, 'W': None, 'b': None}

        mlp_dic['n'] = copy.copy(self.__n)
        mlp_dic['activations'] = copy.copy(self.__activations)
        mlp_dic['type_cost'] = self.__type_cost

        mlp_dic['W'] = []
        for j in range(0, len(self.__weights)):
            mlp_dic['W'].append(self.__weights[j].tolist())

        mlp_dic['b'] = []
        for j in range(0, len(self.__bias)):
            mlp_dic['b'].append(self.__bias[j].tolist())

        # write info in file
        with open('MLP-output.json', 'w') as output_file:
            json.dump(mlp_dic, output_file)

    def load(self, file_name):
        """
        Load trained MLP from file
        :param file_name: file should be in directory of code like: input.json
        :return:
        """
        # read necessary info of MLP from file
        with open(file_name) as json_file:
            mlp_dic = json.load(json_file)

        self.__n = copy.copy(mlp_dic['n'])
        print(self.__n)

        self.__num_layers = len(self.__n)

        self.__activations = copy.copy(mlp_dic['activations'])
        print(self.__activations)

        self.__type_cost = mlp_dic['type_cost']
        print(self.__type_cost)

        # read weights
        self.__weights = []
        for i in range(0, len(mlp_dic['W'])):
            self.__weights.append(np.array(mlp_dic['W'][i]))
            print(self.__weights[len(self.__weights)-1].shape)

        # read biases
        self.__bias = []
        for i in range(0, len(mlp_dic['b'])):
            self.__bias.append(np.array(mlp_dic['b'][i]))
            print(self.__bias[len(self.__bias) - 1].shape)

        self.trained = True
        return self

    def get_number_of_layer(self):
        # return number of layer
        return self.__num_layers

    def get_weights(self):
        # return weights
        return self.__weights

    def get_bias(self):
        # return weights
        return self.__bias

    def get_number_of_neurons(self):
        # return number of neuron in each layer
        return self.__n

    def get_activations_of_layers(self):
        """Return name of activation of each layer"""
        return self.__activations

    def get_type_cost(self):
        return self.__type_cost

###########################################################################
#                                               Test MLP                  #
#                                       functions and their derivative    #
#                                                                         #
###########################################################################
if __name__ == '__main__':

    from read_dataset import read_digit, read_digit_letter

    # read dataset
    #trainX, validX, testX, trainY, validY, testY = read_digit(train_ratio=0.80, valid_ratio=0.10,
    #                                                          test_ratio=0.10, random_state=0)

    trainX, validX, testX, trainY, validY, testY = read_digit_letter(train_ratio=0.80, valid_ratio=0.10,
                                                              test_ratio=0.10, random_state=0)

    # transpose
    trainX = trainX.T
    validX = validX.T
    testX = testX.T
    trainY = trainY.T
    validY = validY.T
    testY = testY.T

    # number of neurons in each layer, first number in number of features, last number is
    # number of classes
    #n = [1024, 20, 20, 10]
    #n = [1024, 80, 20, 10]
    n = [trainX.shape[0], 80, 40, trainY.shape[0]]
    activations = ['tanh', 'tanh','sigmoid']

    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=10000, mode='batch',random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=1.5, max_epoch=100, mode='batch', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=10, mode='momentum', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=1, max_epoch=20, mode='steepest', random_state=0)
    mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=20, mode='steepest', random_state=0)
    ##mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=20, mode='steepest', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=1, max_epoch=10, mode='momentum', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=10, mode='momentum', random_state=2)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=10, mode='stochastic', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=1, mode='stochastic', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=1, mode='stochastic', random_state=0)
    #mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=0.01, max_epoch=30, mode='momentum', random_state=0)#dataful

    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=1000, mode='batch',random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=10000, mode='batch', random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=10, mode='momentum', random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.1, max_epoch=20, mode='steepest', random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=1, max_epoch=10, mode='momentum', random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=10, mode='momentum', random_state=2)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=10, mode='stochastic', random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=1, mode='stochastic', random_state=0)
    # mlp = MLP(n=n, activations=activations, type_of_cost='cross_entropy', learning_rate=0.01, max_epoch=1, mode='stochastic', random_state=0)


    mlp.fit(X=trainX, y=trainY)

    trainY_hat = mlp.predict(trainX)
    train_error = np.sum(np.abs(trainY-trainY_hat))/(trainY.shape[1]*2)
    print("Train Error: {}".format(train_error))

    validY_hat = mlp.predict(validX)
    valid_error = np.sum(np.abs(validY - validY_hat)) / (validY.shape[1] * 2)
    print("Validation Error: {}".format(valid_error))

    testY_hat = mlp.predict(testX)
    test_error = np.sum(np.abs(testY - testY_hat)) / (testY.shape[1] * 2)
    print("Test Error: {}".format(test_error))
