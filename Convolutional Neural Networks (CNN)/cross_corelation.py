import numpy as np


def cross_corr2d(X, K):

    """
        params:

        X : Input 2D Matrix
        K : Kernel/Filter

    """
    h, w = K.shape

    H = X.shape[0]
    W = X.shape[1]

    Y = np.zeros((H - h + 1, W - w + 1))  # Output 2d matrix

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):

            Y[i][j] = np.sum(X[i: i + h, j: j + w] * K)
    return Y

if __name__ == "__main__":

    X = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    K = np.array([[0, 1], [2, 3]])
    print(cross_corr2d(X, K))
