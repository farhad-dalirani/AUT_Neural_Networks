from jordan import *

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from read_dataset import read_dataset
    from errors import mse

    # read dataset, col 0
    col_i = read_dataset(ith_col=5)
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

    # shape of network
    n = [trainX.shape[0], 4, trainY.shape[0]]
    activations = ['tanh' ,'tanh']

    jordan = Jordan(input_size=1, output_size=1, hidden_units=4,
                    activations = ['tanh', 'tanh'], learning_rate=0.01,
                    max_epoch=100, random_state=0)

    #  =======================================
    # load json from file
    jordan.load('2-MLP-output.json')
    #  =======================================

    print('Detail of Jordan that was loaded from file:')
    print('Number of neurons in each layer:', jordan.shape)
    print('activation function in each layer:', jordan.activations)
    print('Feedback Coeficient:', jordan.feedback_coef)


    # plot prediction for training set
    predictions = jordan.predict_series(X=trainX, y=trainY)
    plotting_series(X=trainX, predictions=predictions, target=trainY)

    # plot prediction for validation set
    predictions = jordan.predict_series(X=validX, y=validY)
    plotting_series(X=validX, predictions=predictions, target=validY)

    # plot prediction for training set
    predictions = jordan.predict_series(X=testX, y=testY)
    plotting_series(X=testX, predictions=predictions, target=testY)

    print('\n\n============\nDifferent Cost After Training\n============\n')
    name = ['Training Set', 'Validation Set', 'Test Set']
    for idx, tuple_dataset in enumerate([(trainX, trainY),(validX,validY),(testX,testY)]):
        print('> ', name[idx], ':')
        _x, _y = tuple_dataset
        # calculate different costs for test-set
        sum_of_targets = np.mean(_y, axis=1)[0]
        mse_cost = jordan.mse_cost(_x, _y)
        mae_cost = jordan.mae_cost(_x, _y)
        rmae_cost = mae_cost / sum_of_targets
        PI_cost = jordan.PI_cost(_x, _y)
        print('MSE cost: {}\tMAE cost: {}'.format(mse_cost, mae_cost))
        print('RMAE cost: {}\tPI cost: {}'.format(rmae_cost, PI_cost))

    plt.show()