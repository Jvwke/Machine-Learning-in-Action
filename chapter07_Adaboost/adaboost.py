import numpy as np
from chapter07_Adaboost import decisionStump


def adaboostDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m, n = np.shape(dataArr)
    D = np.mat(np.ones((m, 1)) / m)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = decisionStump.buildStump(dataArr, classLabels, D)
        #print('D: ', D.T)
        alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        #print('classEst: ', classEst.T)
        expon = np.multiply(-1.0 * alpha * np.mat(classLabels).T, classEst)
        D = np.multiply(D, np.exp(expon))
        D = D / D.sum()
        aggClassEst += alpha * classEst
        #print('aggClassEst: ', aggClassEst.T)
        aggErrors = np.multiply(np.sign(aggClassEst) != np.mat(classLabels).T, np.ones((m, 1)))
        errorRate = aggErrors.sum() / m
        #print('total error: ', errorRate)
        if (errorRate == 0.0): break
    return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):
    dataMatrix = np.mat(datToClass)
    m, n = np.shape(dataMatrix)
    aggClassEst = np.mat(np.zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = decisionStump.stumpClasify(dataMatrix,
                                              classifierArr[i]['dim'],
                                              classifierArr[i]['thresh'],
                                              classifierArr[i]['inequal'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
    return np.sign(aggClassEst)


if __name__ == '__main__':
    dataArr, classLabels = decisionStump.loadSimpData()
    weakClassArr, aggClassEst = adaboostDS(dataArr, classLabels)
    print(weakClassArr)
    print(aggClassEst)
    #test
    X = [[0, 0],
         [5, 5]]
    print(adaClassify(X, weakClassArr))