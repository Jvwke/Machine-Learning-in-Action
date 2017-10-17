import numpy as np
import matplotlib.pyplot as plt
from chapter08_Regression import regression

def stageWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regression.regularize(xMat)
    m, n = np.shape(xMat)
    ws = np.zeros((n, 1))
    returnMat = np.zeros((numIt, n))
    for i in range(numIt):
        # print(ws.T)
        lowestError = np.inf
        wsMax = ws.copy()
        for j in range(n):
            for sign in [-1, 1]:
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssE = regression.rssError(yMat.A, yTest.A)
                if rssE < lowestError:
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat


if __name__ == '__main__':
    xArr, yArr = regression.loadData('abalone.txt')
    wsMat = stageWise(xArr, yArr, 0.001, 5000)
    print(wsMat)
    #draw
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(wsMat)
    plt.show()
    #standard regression
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regression.regularize(xMat)
    weights = regression.standRegress(xMat, yMat.T)
    print(weights.T)