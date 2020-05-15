import numpy as np
from mlp import MLP


class Jordan:
    """
    Jordan Class is a wraper for MLP class that I've written before, by
    using mlp class I create a Jordan neural network
    """

    def __init__(self, input_size=1, output_size=1, hidden_units=4,
                 activations = ['tanh', 'tanh'], learning_rate=0.01, max_epoch=200, random_state=0,
                 feedback_coef=1):
        """
        Constructor of Jordan class
        :param input_size: number of features
        :param output_size: size of output vector for a sample
        :param hidden_units: number of hidden layer units
        :param activations: activation functions for different layers
        :param learning_rate: learning rate
        :param max_epoch:
        :param random_state:
        :param feedback_coef: coefficient of feedback signal
        """
        # create MLP
        n = [input_size+1, hidden_units, output_size]
        self.mlp = MLP(n=n, activations=activations, type_of_cost='MSE', learning_rate=learning_rate, max_epoch=max_epoch,
                  mode='stochastic', random_state=random_state)
        self.__max_epochs = max_epoch
        self.feedback_coef = feedback_coef
        self.shape = [input_size, hidden_units, output_size]
        self.activations = activations

    def forward_pass_a_sample(self, input, feedback=0):
        """

        :param input: input sample, scalar
        :param feedback: output of previous sample in series, scalar
        :return:
        """
        # concat input and feedback
        input_feedback = np.array([input, feedback])
        input_feedback = np.reshape(input_feedback, (2,1))

        catch_I, catch_Y = self.mlp.forward_propagation(input_feedback)

        return catch_I, catch_Y


    def mse_cost(self, X, y):
        """
        Calculate MSE cost for predictions of a series
        :param X: training dataset, each column is an observation
        :param y: target of training dataset
        :return:
        """
        # cost function at begining
        cost = 0
        feedback = 0
        for i_sample in range(0, X.shape[1]):
            # calculate output of network fo a sample
            catch_I, catch_Y = self.forward_pass_a_sample(input=X[0, i_sample], feedback=feedback)

            # new feedback
            feedback = catch_Y[-1][0, 0]
            cost += (feedback - y[0, i_sample]) ** 2
            feedback = feedback * self.feedback_coef
        cost /= X.shape[1]
        return cost

    def mae_cost(self, X, y):
        """
        Calculate MAE cost for predictions of a series
        :param X: training dataset, each column is an observation
        :param y: target of training dataset
        :return:
        """
        # cost function at begining
        cost = 0
        feedback = 0
        for i_sample in range(0, X.shape[1]):
            # calculate output of network fo a sample
            catch_I, catch_Y = self.forward_pass_a_sample(input=X[0, i_sample], feedback=feedback)

            # new feedback
            feedback = catch_Y[-1][0, 0]
            cost += abs(feedback - y[0, i_sample])
            feedback = feedback * self.feedback_coef
        cost /= X.shape[1]
        return cost

    def PI_cost(self, X, y):
        """
        Calculate PI cost for predictions of a series
        :param X: training dataset, each column is an observation
        :param y: target of training dataset
        :return:
        """
        # cost function at begining
        cost_num = 0
        cost_denum = 0
        feedback = 0
        for i_sample in range(0, X.shape[1]):
            # calculate output of network fo a sample
            catch_I, catch_Y = self.forward_pass_a_sample(input=X[0, i_sample], feedback=feedback)

            # new feedback
            feedback = catch_Y[-1][0, 0]
            cost_num += (feedback - y[0, i_sample])**2
            if i_sample > 0:
                cost_denum += (y[0, i_sample] - y[0, i_sample-1])**2
            feedback = feedback * self.feedback_coef
        cost = 1-(cost_num / cost_denum)
        return cost

    def predict_series(self, X, y):
        """
        Calculate output of give input series X
        :param X: training dataset, each column is an observation
        :param y: target of training dataset
        :return:
        """
        output = []

        feedback = 0
        for i_sample in range(0, X.shape[1]):
            # calculate output of network fo a sample
            catch_I, catch_Y = self.forward_pass_a_sample(input=X[0, i_sample], feedback=feedback)

            # new feedback
            feedback = catch_Y[-1][0, 0]
            output.append(feedback)
            feedback = feedback * self.feedback_coef

        output = np.array(output)
        output = np.reshape(output, (1, output.shape[0]))
        return output

    def fit(self, X, y, validX=None, validy=None):
        """
        Learn learnabble parameters by using back propagation
        :param X: training dataset, each column is an observation
        :param y: target of training dataset
        :param X: validation dataset, each column is an observation
        :param y: target of validation dataset
        :return:
        """

        # list of different costs
        mse_list = []
        mae_list = []
        rmae_list = []
        PI_list = []

        if not(validX is None):
            # list of different costs for validation
            mse_list_valid = []
            mae_list_valid = []
            rmae_list_valid = []
            PI_list_valid = []
            sum_of_targets_valid = np.mean(validy, axis=1)[0]

        # cost function at begining
        sum_of_targets = np.mean(y, axis=1)[0]
        init_msecost = self.mse_cost(X, y)
        init_maecost = self.mae_cost(X, y)
        init_rmaecost = init_maecost/sum_of_targets
        print('iter {}/{}\nMSE cost: {}\tMAE cost: {}'.format(0, self.__max_epochs - 1, init_msecost, init_maecost))
        print('RMAE cost: {}\tPI cost: -'.format(init_rmaecost))

        # for each epoch, give all sample one by one to network and update weights
        for i_epoch in range(0,  self.__max_epochs):

            # give samples to network one by one
            feedback = 0
            for i_sample in range(0, X.shape[1]):

                # calculate output of network fo a sample
                catch_I, catch_Y = self.forward_pass_a_sample(input=X[0, i_sample], feedback=feedback)

                # new feedback
                feedback = catch_Y[-1][0, 0]
                feedback = feedback * self.feedback_coef

                # calculate gradients
                grad_w, grad_b = self.mlp.backward_propagation_stochastic(target=y[0,i_sample], catch_I=catch_I, catch_Y=catch_Y)

                # update weights
                self.mlp.update_weight_bias_normal(grad_w=grad_w, grad_b=grad_b)

            # calculate different cost
            mse_cost = self.mse_cost(X, y)
            mse_list.append(mse_cost)
            mae_cost = self.mae_cost(X, y)
            mae_list.append(mae_cost)
            rmae_cost = mae_cost/sum_of_targets
            rmae_list.append(rmae_cost)
            PI_cost = self.PI_cost(X, y)
            PI_list.append(PI_cost)
            print('===================\niter {}/{}\nTraining set\nMSE cost: {}\tMAE cost: {}'.format(i_epoch+1, self.__max_epochs, mse_cost, mae_cost))
            print('RMAE cost: {}\tPI cost: {}'.format(rmae_cost, PI_cost))

            if not(validX is None):
                mse_cost = self.mse_cost(validX, validy)
                mse_list_valid.append(mse_cost)
                mae_cost = self.mae_cost(validX, validy)
                mae_list_valid.append(mae_cost)
                rmae_cost = mae_cost / sum_of_targets_valid
                rmae_list_valid.append(rmae_cost)
                PI_cost = self.PI_cost(validX, validy)
                PI_list_valid.append(PI_cost)
                print(
                    'Validation Set\nMSE cost: {}\tMAE cost: {}'.format(mse_cost, mae_cost))
                print('RMAE cost: {}\tPI cost: {}'.format(rmae_cost, PI_cost))

        # save model
        self.save()

        if validX is None:
            # return cost for training set during training
            return mse_list, mae_list, rmae_list, PI_list
        else:
            # return cost for training set during training
            return mse_list, mae_list, rmae_list, PI_list, mse_list_valid, mae_list_valid, rmae_list_valid, PI_list_valid

    def save(self):
        """
        save jordan network in two files.
        MLP-output.json and extra-MLP-output.json
        :return:
        """
        import json

        # save underlying mlp
        self.mlp.save()

        dic = {'feedback_cof':self.feedback_coef, 'max_epoch':self.__max_epochs,
               'layers':self.shape, 'activations':self.activations}
        #save some parameters
        # write info in file
        with open('extra-MLP-output.json', 'w') as output_file:
            json.dump(dic, output_file)

    def load(self, file_name):
        """
        Load stored network
        :param self:
        :param file_name: file should be in directory of code like: input.json
        :return:
        """
        import json
        self.mlp.load(file_name=file_name)

        # read necessary and extra info of jordan from file
        with open('extra-'+file_name) as json_file:
            jordan_dic = json.load(json_file)

        dic = {'feedback_cof': self.feedback_coef, 'max_epoch': self.__max_epochs,
               'layers': self.shape, 'activations': self.activations}
        self.feedback_coef = jordan_dic['feedback_cof']
        self.__max_epochs = jordan_dic['max_epoch']
        self.shape = jordan_dic['layers']
        self.activations = jordan_dic['feedback_cof']


def plotting_series(X, predictions, target, figsize=(10, 3), name=""):
    """ Plot Series
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=3, sharex=True, figsize=figsize)
    fig.suptitle(name)
    ax[0].plot(X.flatten(), label='inputs')
    ax[0].legend()
    ax[1].plot(target.flatten(), label='targets')
    ax[1].plot(predictions.flatten(), '--', markersize=0.1, alpha=1, label='preds')
    ax[1].legend()
    ax[2].plot((target-predictions).flatten(), label='difference of targets and predictions')
    ax[2].legend()
    ax[2].set_xlim([0, target.shape[1]-1])
    ax[2].set_ylim([-0.3, 0.3])



def plotting_errors_of_training(mse_list, mae_list, rmae_list, PI_list, figsize=(10, 10), name=""):
    """ Plot error of prediction for training and validation set
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=figsize)
    fig.suptitle('Costs During Training - '+name)
    ax[0, 0].plot(mse_list, label='MSE')
    ax[0, 0].legend()
    ax[0, 1].plot(mae_list, label='MAE')
    ax[0, 1].legend()
    ax[1, 0].plot(rmae_list, label='RMAE')
    ax[1, 0].legend()
    ax[1, 1].plot(PI_list, label='PI')
    ax[1, 1].legend()
###########################################################################
#                                               Test jordan               #
#                                                                         #
###########################################################################
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from read_dataset import read_dataset
    from errors import mse

    # read dataset
    col_i = read_dataset(ith_col=0)
    trainX = col_i[0:10000, :]
    validX = col_i[10000:13000, :]
    testX = col_i[13000:23000, :]
    trainY = col_i[1:10001, :]
    validY = col_i[10001:13001, :]
    testY = col_i[13001:23001, :]

    # transpose
    trainX = trainX.T
    validX = validX.T
    testX = testX.T
    trainY = trainY.T
    validY = validY.T
    testY = testY.T

    #print(trainX.shape, trainY.shape)
    #print(trainX[0,0])

    # number of neurons in each layer, first number in number of features, last number is
    # number of classes
    #n = [1024, 20, 20, 10]
    #n = [1024, 80, 20, 10]
    n = [trainX.shape[0], 4, trainY.shape[0]]
    activations = ['tanh' ,'tanh']

    jordan = Jordan(input_size=1, output_size=1, hidden_units=4,
                    activations = ['tanh', 'tanh'], learning_rate=0.01,
                    max_epoch=100, random_state=0)

    # train jordan network
    mse_list, mae_list, rmae_list, PI_list, mse_list_valid,\
    mae_list_valid, rmae_list_valid, PI_list_valid =\
        jordan.fit(X=trainX, y=trainY, validX=validX, validy=validY)

    # plot errors during training
    plotting_errors_of_training(mse_list=mse_list, mae_list=mae_list,
                                rmae_list=rmae_list, PI_list=PI_list, name='Training Set')
    plotting_errors_of_training(mse_list=mse_list_valid, mae_list=mae_list_valid,
                                rmae_list=rmae_list_valid, PI_list=PI_list_valid, name='Validation Set')

    # plot prediction for training set
    predictions = jordan.predict_series(X=trainX, y=trainY)
    plotting_series(X=trainX, predictions=predictions, target=trainY)

    # plot prediction for validation set
    predictions = jordan.predict_series(X=validX, y=validY)
    plotting_series(X=validX, predictions=predictions, target=validY)

    # plot prediction for training set
    predictions = jordan.predict_series(X=testX, y=testY)
    plotting_series(X=testX, predictions=predictions, target=testY)

    plt.show()