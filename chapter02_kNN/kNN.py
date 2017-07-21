import numpy as np
import matplotlib.pyplot as plt
import operator
import os

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0.0, 0.0], [0.0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(X, dataSet, labels, k):
    diffMat = X - dataSet
    sqDiffMat = diffMat**2
    sqDistances = np.sum(sqDiffMat, axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    text = fr.readlines()
    n = len(text)
    X = np.zeros((n, 3))
    y = []
    index = 0
    for line in text:
        line = line.strip()
        listFromLine = line.split('\t')
        X[index, :] = listFromLine[0:3]
        y.append(int(listFromLine[-1]))
        index += 1
    return X, y

def autoNorm(X):
    minVals = np.min(X, axis=0)
    maxVals = np.max(X, axis=0)
    ranges = maxVals - minVals
    normX = (X - minVals) / ranges
    return normX, ranges, minVals

def datingClassTest():
    X, y = file2matrix('datingTestSet2.txt')
    X, ranges, minVals = autoNorm(X)
    p = 0.1
    n = X.shape[0]
    testSize = int(n * p)
    errorCount = 0.0
    for i in range(testSize):
        res = classify0(X[i, :], X[testSize:n, :], y[testSize:n], 3)
        print('the classifier came back with %d, the real answer is: %d' % (res, y[i]))
        if (res != y[i]): errorCount += 1.0
    print('the total error rate is: %f' % (errorCount / float(testSize)))

def classifyPerson():
    X, y = file2matrix('datingTestSet2.txt')
    X, ranges, minVals = autoNorm(X)
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice Cream consumed per year?'))
    inArr = np.array([ffMiles, percentTats, iceCream])
    res = classify0((inArr - minVals) / ranges, X, y, 3)
    print('You will probably like this person:', resultList[res - 1])

def img2vec(filename):
    vec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        line = fr.readline()
        for j in range(32):
            vec[0, 32 * i + j] = int(line[j])
    return vec

def handwritingClassTest():
    hwLabels = []
    trainingFileList = os.listdir('digits/trainingDigits')
    n = len(trainingFileList)
    trainingMat = np.zeros((n, 1024))
    for i in range(n):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vec('digits/trainingDigits/%s' % fileNameStr)

    testFileList = os.listdir('digits/testDigits')
    errorCount = 0.0
    nTest = len(testFileList)
    for i in range(nTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vec('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with: %d. the real answer is: %d' % (classifierResult, classNumStr))
        if (classifierResult != classNumStr):
            errorCount += 1.0
            # print('filename: %s %d' % (fileNameStr, classifierResult))

    print('the total number of errors is: %d' % errorCount)
    print('the total error rate is: %f' % (errorCount / float(nTest)))

def draw():
    X, y = file2matrix('datingTestSet2.txt')
    X, ranges, minVals = autoNorm(X)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(X[:, 0], X[:, 1], 15.0 * np.array(y), 15.0 * np.array(y))
    plt.show()

if __name__ == '__main__':
    datingClassTest()
    classifyPerson()
    handwritingClassTest()

