import numpy as np
from scipy.misc import comb


def purity_measure(clusters, classes):
    """
    This function calculate the purity for given cluster and real classes value
    :param clusters: the cluster assignments array
    :param classes: the ground truth classes
    :returns: the purity score
    """
    #print(clusters.shape)
    #print(type(clusters))
    #print(type(clusters[0]))
    #print(classes.shape)
    #print(type(classes))
    #print(type(classes[0]))

    # zip predicted clusters and real class values
    cluster_class = np.hstack((clusters, classes))

    n_accurate = 0.0

    for cluster in np.unique(cluster_class[:, 0]):
        z = cluster_class[cluster_class[:, 0] == cluster, 1]
        x = np.argmax(np.bincount(z))
        n_accurate += len(z[z == x])
        #print('cluster {}- class {}-rep {}/{}'.format(cluster, x, len(z[z==x]), len(z)))

    return n_accurate / cluster_class.shape[0]


def rand_index(clusters, classes):
    """
    Rand Index Measure
    :param clusters: the cluster assignments array
    :param classes: the ground truth classes
    :return: the RI score
    """

    TP_FP = comb(np.bincount(clusters[:, 0]), 2).sum()
    TP_FN = comb(np.bincount(classes[:, 0]), 2).sum()

    cluster_class = np.hstack((clusters, classes))

    TP = sum(comb(np.bincount(cluster_class[cluster_class[:, 0] == i, 1]), 2).sum() for i in set(clusters[:, 0]))
    FP = TP_FP - TP
    FN = TP_FN - TP
    TN = comb(len(cluster_class), 2) - TP - FP - FN
    RI = (TP + TN) / (TP + FP + FN + TN)

    return RI


def f_measure(clusters, classes):
    """
    f Measure
    :param clusters: the cluster assignments array
    :param classes: the ground truth classes
    :return: the f score
    """
    #print('>>>')
    #print(np.bincount(clusters[:, 0]).shape)
    TP_FP = comb(np.bincount(clusters[:, 0]), 2).sum()
    TP_FN = comb(np.bincount(classes[:, 0]), 2).sum()

    cluster_class = np.hstack((clusters, classes))
    #print('===> ',cluster_class[cluster_class[:, 0] == 1, 1].reshape(1, cluster_class[cluster_class[:, 0] == 1, 1].shape[0]))
    TP = sum(comb(np.bincount(cluster_class[cluster_class[:, 0] == i, 1]), 2).sum() for i in set(clusters[:, 0]))
    FP = TP_FP - TP
    FN = TP_FN - TP
    #TN = comb(len(cluster_class), 2) - TP - FP - FN
    P = TP/(TP + FP)
    R = TP/(TP + FN)
    # f-measure
    F_score = 2 * (P * R) / (P + R)
    return F_score

