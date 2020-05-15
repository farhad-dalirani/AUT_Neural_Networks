def plotting(mlp_object, trainX, trainY, validX, validY):
    """
    This function should be run inside a thread during training MLP
    Plot:
        sum of weights in each epoch
        plot cost function in each epoch
        plot train error in each epoch
        plot validation error in each epoch
    :param mlp_object: an object of class MLP
    :param trainX: training data each column is as observation
    :param trainY: training data label each column is an one hot label
    :param validX: validation data each column is as observation
    :param validY: validation data label each column is an one hot label
    :return:
    """
    import numpy as np
    import time
    import matplotlib.pyplot as plt

    # colors for plotting
    colors = [plt.cm.jet(i) for i in np.linspace(0, 1, 5)]

    sum_of_weights_per_epochs = []
    sum_of_error_layer_per_epochs = {i:[] for i in range(0, mlp_object.get_number_of_layer()-1)}
    train_errors = []
    validation_errors = []

    if mlp_object.plot_info == False:
        raise ValueError('MLP plotting attribute shouldn\'t be False')

    # create plots
    fig_sum_weights = plt.figure()
    plt.title('Sum of weights and biases in each epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Sum of weights and biases')
    plt.ion()

    fig_cost = plt.figure()
    plt.title('cost function in each epoch')
    plt.xlabel('Epochs')
    plt.ylabel('{} cost'.format(mlp_object.get_type_cost()))
    plt.ion()

    fig_train_error = plt.figure()
    plt.title('Train set error in each epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.ion()

    fig_valid_error = plt.figure()
    plt.title('Validation set error in each epoch')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.ion()

    if mlp_object.dI_epochs is not None:
        fig_error_each_layer = plt.figure()
        plt.title('Sum of propagated error for each layer (dE/dI)')
        plt.xlabel('Epochs')
        plt.ylabel('Sum of propagated (dE/dI)')
        plt.ion()
        colors_grad = [plt.cm.jet(i) for i in np.linspace(0, 1, mlp_object.get_number_of_layer()-1)]

    extra_iter = 2
    first_time = True
    while mlp_object.trained == False or extra_iter >= 0:

        if mlp_object.trained == True:
            extra_iter = extra_iter - 1

        while len(mlp_object.cost_epochs) != len(mlp_object.weigths_epochs):
            time.sleep(0.01)
        time.sleep(0.01)

        l1 = len(mlp_object.cost_epochs)
        l2 = len(mlp_object.weigths_epochs)

        while l1 > 0 and l1==l2 and l1 != len(sum_of_weights_per_epochs):
            index = len(sum_of_weights_per_epochs)#-1

            # add sum of weights and bias
            sum = 0
            for i in range(0, len(mlp_object.weigths_epochs[index])):
                sum = sum + np.sum(mlp_object.weigths_epochs[index][i])
                sum = sum + np.sum(mlp_object.biases_epochs[index][i])

            sum_of_weights_per_epochs.append(sum)

            if mlp_object.dI_epochs is not None:
                #print('++++++>', index)
                #print('======>', len(mlp_object.dI_epochs))
                #time.sleep(0.300)
                for i in range(0, len(mlp_object.dI_epochs[index])):
                    #sum = np.sum(np.abs(mlp_object.dI_epochs[index][i]))
                    sum = np.sum(mlp_object.dI_epochs[index][i])
                    sum_of_error_layer_per_epochs[i].append(sum)

            # calculate train and validation error
            trainY_hat = mlp_object.predict(trainX)
            train_error = np.sum(np.abs(trainY - trainY_hat)) / (trainY.shape[1] * 2)
            train_errors.append(train_error)

            validY_hat = mlp_object.predict(validX)
            valid_error = np.sum(np.abs(validY - validY_hat)) / (validY.shape[1] * 2)
            validation_errors.append(valid_error)


        # plot data
        plt.figure(fig_sum_weights.number)
        plt.plot(range(1, len(sum_of_weights_per_epochs) + 1), sum_of_weights_per_epochs, color=colors[0],marker='o', markersize=3)
        plt.pause(0.05)

        plt.figure(fig_cost.number)
        plt.plot(range(1, len(mlp_object.cost_epochs) + 1), mlp_object.cost_epochs, color=colors[1], marker='o', markersize=3)
        plt.pause(0.05)

        plt.figure(fig_train_error.number)
        plt.plot(range(1, len(train_errors) + 1), train_errors, color=colors[2], marker='o', markersize=3)
        plt.pause(0.05)

        plt.figure(fig_valid_error.number)
        plt.plot(range(1, len(validation_errors) + 1), validation_errors, color=colors[3], marker='o', markersize=3)
        plt.pause(0.05)

        if mlp_object.dI_epochs is not None:
            plt.figure(fig_error_each_layer.number)
            for key in sum_of_error_layer_per_epochs:
                plt.plot(range(1, len(sum_of_error_layer_per_epochs[key])), sum_of_error_layer_per_epochs[key][1::],
                         color=colors_grad[key % 15], marker='o', markersize=3,label='layer{0}'.format(key+2) if first_time==True else '')
            first_time = False
            plt.legend()
            plt.pause(0.05)

    plt.show(block=True)
