from jordan import *

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from read_dataset import read_dataset
    from errors import mse

    # read dataset, col 0, one step
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

    # net 1
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=1,
    #                activations = ['tanh', 'tanh'], learning_rate=0.01,
    #                max_epoch=100, random_state=0)
    # net 2
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #               activations = ['tanh', 'tanh'], learning_rate=0.001,
    #                max_epoch=150, random_state=0)
    # net 3
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=3,
    #                activations=['tanh', 'tanh'], learning_rate=0.001,
    #                max_epoch=180, random_state=0)
    # net 4
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=4,
    #                activations=['tanh', 'tanh'], learning_rate=0.001,
    #                max_epoch=180, random_state=0)
    # net 6
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #                activations = ['tanh', 'tanh'], learning_rate=0.001,
    #                max_epoch=150, random_state=0, feedback_coef=0.01)
    # net 7
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #                activations=['tanh', 'tanh'], learning_rate=0.001,
    #                max_epoch=150, random_state=0, feedback_coef=0.0001)
    # net 8
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #                activations=['tanh', 'tanh'], learning_rate=0.001,
    #                max_epoch=150, random_state=0, feedback_coef=10)
    # net 9
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #                activations = ['tanh', 'tanh'], learning_rate=0.01,
    #                max_epoch=150, random_state=0)
    # net 10
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #                activations = ['tanh', 'tanh'], learning_rate=0.00001,
    #                max_epoch=150, random_state=0)
    # net 11
    #jordan = Jordan(input_size=1, output_size=1, hidden_units=2,
    #               activations = ['tanh', 'linear'], learning_rate=0.000001,
    #                max_epoch=150, random_state=0)


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
    plotting_series(X=trainX, predictions=predictions, target=trainY, name='Training Set')

    # plot prediction for validation set
    predictions = jordan.predict_series(X=validX, y=validY)
    plotting_series(X=validX, predictions=predictions, target=validY, name='Validation Set')

    # plot prediction for training set
    predictions = jordan.predict_series(X=testX, y=testY)
    plotting_series(X=testX, predictions=predictions, target=testY, name='Test Set')

    print('\n\n============\nDifferent Costs After Training\n============\n')
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