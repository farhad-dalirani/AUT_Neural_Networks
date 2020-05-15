import numpy as np


def mse(target, y):
    """
    Mean Square Error
    :param target: number of samples * 1, numpy array
    :param y: number of samples * 1, numpy array
    :return: score
    """
    mse_score = ((target-y)**2).mean(axis=1)

    return mse_score[0]