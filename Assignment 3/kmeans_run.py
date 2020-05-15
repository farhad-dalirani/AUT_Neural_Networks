if __name__ == '__main__':

    # Import packages
    import numpy as np
    from measures import purity_measure, rand_index, f_measure
    import time
    import copy
    import sys
    from read_dataset import read_dataset
    import json
    import threading
    from sklearn.cluster import KMeans

    # read dataset
    print('reading dataset ...\n')
    trainX, validX, testX, trainY, validY, testY = read_dataset(train_ratio=0.80, valid_ratio=0.10,
                                                                  test_ratio=0.10, random_state=0)

    # for different number of clusters
    for k in [2, 3, 4, 9, 12, 16, 20, 24, 27, 36, 64]:
        # Kmeans clustering
        kmeans = KMeans(n_clusters=k, random_state=0).fit(trainX)

        y_train_pre = kmeans.predict(X=trainX)
        y_valid_pre = kmeans.predict(X=validX)
        y_test_pre = kmeans.predict(X=testX)

        y_train_pre = y_train_pre.reshape(y_train_pre.shape[0], 1)
        y_valid_pre = y_valid_pre.reshape(y_valid_pre.shape[0], 1)
        y_test_pre = y_test_pre.reshape(y_test_pre.shape[0], 1)

        print('==================================================================')
        print('K={}'.format(k))
        print('PU-Train-Kmeans: ', purity_measure(clusters=y_train_pre, classes=trainY))
        print('PU-Valid-Kmeans: ', purity_measure(clusters=y_valid_pre, classes=validY))
        print('PU-Test-Kmeans: ', purity_measure(clusters=y_test_pre, classes=testY))

        print('RI-Train-Kmeans: ', rand_index(clusters=y_train_pre, classes=trainY))
        print('RI-Valid-Kmeans: ', rand_index(clusters=y_valid_pre, classes=validY))
        print('RI-Test-Kmeans: ', rand_index(clusters=y_test_pre, classes=testY))

        print('F-Measure-Train-Kmeans: ', f_measure(clusters=y_train_pre, classes=trainY))
        print('F-Measure-Valid-Kmeans: ', f_measure(clusters=y_valid_pre, classes=validY))
        print('F-Measure-Test-Kmeans: ', f_measure(clusters=y_test_pre, classes=testY))