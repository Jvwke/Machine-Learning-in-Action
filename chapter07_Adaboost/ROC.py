import numpy as np
import matplotlib.pyplot as plt
from chapter07_Adaboost import adaboost
from chapter07_Adaboost import adaColic


def plotROC(predStrengths, classLabels):
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClass = np.sum(np.array(classLabels) == 1.0)
    yStep = 1 / numPosClass
    xStep = 1 / (len(classLabels) - numPosClass)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0.0
            delY = yStep
        else:
            delX = xStep
            delY = 0.0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    ax.axis([0, 1, 0, 1])
    print('AUC:', ySum * xStep)
    plt.show()


if __name__ == '__main__':
    dataArr, LabelArr = adaColic.loadDataSet('horseColicTraining2.txt')
    weakClassArr, aggClassEst = adaboost.adaboostDS(dataArr, LabelArr, 10)
    plotROC(aggClassEst.T, LabelArr) # AUC: 0.858296963506


