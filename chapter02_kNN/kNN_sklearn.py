import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN


def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % (fileNameStr))

    knn = KNN(n_neighbors=3, algorithm='auto')
    knn.fit(trainingMat, hwLabels)

    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % (fileNameStr))
        classifierResult = knn.predict(vectorUnderTest)
        print('the classifier came back with: %d. the real answer is: %d' %
              (classifierResult, classNumber))
        if (classifierResult != classNumber):
            errorCount += 1.0
    print("error count: %d\nerror rate: %f%%" % (errorCount, errorCount / mTest * 100.0))


if __name__ == '__main__':
    handwritingClassTest()
