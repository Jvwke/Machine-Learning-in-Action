from chapter05_LogisticRegression import logRegres
import numpy as np

def classifyVector(inX, weights):
    prob = logRegres.sigmoid(np.sum(inX * weights))
    return 1.0 if prob > 0.5 else 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt', 'r')
    frTest = open('horseColicTest.txt', 'r')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = logRegres.stocGradAscent1(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21): lineArr.append(float(currLine[i]))
        if int(classifyVector(lineArr, trainWeights)) != int(currLine[21]): errorCount += 1
    errorRate = float(errorCount) / numTestVec
    print('the error rate of this test is: %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests): errorSum += colicTest()
    print('after %d iterations the average error rate is: %f' % (numTests, errorSum / float(numTests)))

if __name__ == '__main__':
    multiTest()