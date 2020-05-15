##################################################################
# Investing correctness of implemented SOM with iris dataset     #
##################################################################

if __name__ == '__main__':

    from SOM import SOM
    import numpy as np
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from measures import purity_measure, rand_index, f_measure
    from sklearn.cluster import KMeans

    # Read Iris dataset
    iris = datasets.load_iris()

    # Use features petal length, petal width
    X = iris.data
    y = iris.target

    # Split data to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)
    y_train = y_train.reshape((X_train.shape[0], 1))
    y_test = y_test.reshape((X_test.shape[0], 1))

    ############################
    # SOM
    ############################
    # select some samples as random weights fo initialization
    initial_sample_indexes = np.random.permutation(X_train.shape[0])

    shape_net = (1, 3, 1)
    number_of_neurons = shape_net[0] * shape_net[1] * shape_net[2]

    som = SOM(shape=shape_net,
              number_of_feature=X_train.shape[1],
              distance_measure_str='cosine',
              topology='cubic',
              init_learning_rate=0.4,
              max_epoch=300,
              samples_for_init=X_train[initial_sample_indexes[0:number_of_neurons]])

    # Train SOM
    som.fit(X=X_train)

    y_train_pre = som.predict(X=X_train)
    y_test_pre = som.predict(X=X_test)

    print("###############################\n###########  SOM  #############\n###############################")
    print('PU-Train-SOM: ', purity_measure(clusters=y_train_pre, classes=y_train))
    print('PU-Test-SOM: ', purity_measure(clusters=y_test_pre, classes=y_test))

    print('RI-Train-SOM: ', rand_index(clusters=y_train_pre, classes=y_train))
    print('RI-Test-SOM: ', rand_index(clusters=y_test_pre, classes=y_test))

    print('F Measure-SOM: ', f_measure(clusters=y_train_pre, classes=y_train))
    print('F Measure-SOM: ', f_measure(clusters=y_test_pre, classes=y_test))

    ############################
    # kmeans
    ############################
    print("###############################\n##########  Kmean  ############\n###############################")
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
    y_train_pre = kmeans.predict(X_train)
    y_train_pre = y_train_pre.reshape(y_train_pre.shape[0],1)
    y_test_pre = kmeans.predict(X_test)
    y_test_pre = y_test_pre.reshape(y_test_pre.shape[0], 1)

    print('PU Train-Kmean: ', purity_measure(clusters=y_train_pre, classes=y_train))
    print('PU Test-Kmean: ', purity_measure(clusters=y_test_pre, classes=y_test))
    print('PU Test-Kmean: ', purity_measure(clusters=y_test_pre, classes=y_test))

    print('RI Train-Kmean: ', rand_index(clusters=y_train_pre, classes=y_train))
    print('RI Test-Kmean: ', rand_index(clusters=y_test_pre, classes=y_test))

    print('F Measure Train-Kmean: ', f_measure(clusters=y_train_pre, classes=y_train))
    print('F Measure Test-Kmean: ', f_measure(clusters=y_test_pre, classes=y_test))
