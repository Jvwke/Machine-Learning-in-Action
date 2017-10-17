import numpy as np
from chapter08_Regression import regression
from chapter08_Regression import LWLR

if __name__ == '__main__':
    abX, abY = regression.loadData('abalone.txt')
    for k in [0.1, 1, 10]:
        print('parameter k =', k)
        yHatTrain = LWLR.lwlrTest(abX[0:99], abX[0:99], abY[0:99], k)
        errorTrain = regression.rssError(abY[0:99], yHatTrain.T)
        print('training error:', errorTrain)
        yHatTest = LWLR.lwlrTest(abX[100:199], abX[0:99], abY[0:99], k)
        errorTest = regression.rssError(abY[100:199], yHatTest)
        print('test error:', errorTest)
    ws = regression.standRegress(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    error = regression.rssError(abY[100:199], yHat.T.A)
    print('standard regression error:', error)