import numpy as np
import matplotlib.pyplot as plt
from chapter08_Regression import regression


def ridgeRegres(xMat, yMat, lam=0.2):
    xTx = xMat.T * xMat
    m, n = np.shape(xMat)
    denom = xTx + np.eye(n) * lam
    if np.linalg.det(denom) == 0.0:
        print('Error: this matrix is singular, cannot do inverse')
        return
    ws = denom.I * xMat.T * yMat
    return ws


def ridgeTest(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat, 0)
    xVar = np.var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, np.exp(i-10))
        wMat[i, :] = ws.T
    return wMat


if __name__ == '__main__':
    abX, abY = regression.loadData('abalone.txt')
    ridgeWeights = ridgeTest(abX, abY)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(ridgeWeights)
    plt.show()