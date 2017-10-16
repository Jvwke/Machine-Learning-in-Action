import numpy as np
import matplotlib.pyplot as plt


def loadData(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def standRegress(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print('Error: This matrix is singular, cannot do inverse!')
        return
    ws = xTx.I * xMat.T * yMat
    return ws


def draw(xArr, yArr, ws):
    xMat = np.mat(xArr); yMat = np.mat(yArr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # matrix.A	Return self as an ndarray object.
    ax.scatter(xMat[:, 1].flatten().A[0], yMat.T[:, 0].flatten().A[0])
    plt.xlabel('x0'); plt.ylabel('x1')
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    print('correlation coefficients of yHat and yMat:\n', np.corrcoef(yHat.T, yMat))
    ax.plot(xCopy[:, 1], yHat)
    plt.show()

if __name__ == '__main__':
    xArr, yArr = loadData('ex0.txt')
    ws = standRegress(xArr, yArr)
    print('weights:\n', ws)
    draw(xArr, yArr, ws)
