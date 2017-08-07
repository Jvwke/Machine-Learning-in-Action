import numpy as np

def loadDatsSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.array(dataMatIn)
    m, n = dataMatrix.shape
    labelMat = np.array(classLabels).reshape(m, 1)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix.dot(weights))
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T.dot(error)
    return weights

if __name__ == '__main__':
    dataArr, labelMat = loadDatsSet()
    weights = gradAscent(dataArr, labelMat)
    print(weights)
