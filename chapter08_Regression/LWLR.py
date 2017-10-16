import numpy as np
import matplotlib.pyplot as plt
from chapter08_Regression import  regression

def lwlr(testPoint, xArr, yArr, k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m, n = np.shape(xMat)
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T / (-2.0 * k**2))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print('Error: This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m, n = np.shape(testArr)
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def draw(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr)
    srtInd = xMat[:, 1].argsort(0)
    xSort = xMat[srtInd][:, 0, :]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xSort[:, 1], yHat[srtInd])
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T.flatten().A[0], s=2, c='red')
    plt.show()

if __name__ == '__main__':
    xArr, yArr = regression.loadData('ex0.txt')
    yHat = lwlrTest(xArr, xArr, yArr, 0.003)
    print(yHat)
    draw(xArr, yArr)


