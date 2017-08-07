import numpy as np
import feedparser
import operator
import re

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setofWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word: %s is not in my vocabulary!', word)
    return returnVec

def bagofWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
        else:
            print('the word: %s is not in my vocabulary!' % word)
    return returnVec

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = np.sum(trainCategory) / float(numTrainDocs)

    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # p0Denom = 0.0
    # p1Denom = 0.0

    # 拉普拉斯平滑，见《统计学习方法》P51
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num +=trainMatrix[i]
            p1Denom += np.sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += np.sum(trainMatrix[i])
    # p0Vect = p0Num / p0Denom
    # p1Vect = p1Num / p1Denom
    p0Vect = np.log(p0Num / p0Denom)
    p1Vect = np.log(p1Num / p1Denom)
    return p0Vect, p1Vect, pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = np.sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = np.sum(vec2Classify * p0Vec) + np.log(1 - pClass1)
    return 1 if p1 > p0 else 0

def testingNB(vec2Classify):
    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    thisDoc = np.array(setofWords2Vec(myVocabList, vec2Classify))
    print(vec2Classify, 'classified as ', classifyNB(thisDoc, p0V, p1V, pAb))

def textParse(bigString):
    listOfTokens = re.split('\\W*', bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = np.random.randint(0, len(trainingSet))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setofWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))

    errorCount = 0
    for docIndex in testSet:
        wordVector = setofWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print('the error document is:', docList[docIndex])
    print('the error rate is :', float(errorCount) / len(testSet))

def calcMostFreq(vocabList, fullText):
    freqDict = {}
    for token in vocabList:
        freqDict[token] = fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedFreq[:30]

def localWords(feed1, feed0):
    docList = []
    classList = []
    fullText = []
    minLen = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList, fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = list(range(2 * minLen))
    testSet = []
    for i in range(20):
        randIndex = np.random.randint(0, len(trainingSet))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagofWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagofWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is: ', float(errorCount) / len(testSet))
    return vocabList, p0V, p1V

def getTopWords(ny, sf):
    vocabList, p0V, p1V = localWords(ny, sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -6.0 : topSF.append((vocabList[i], p0V[i]))
        if p1V[i] > -6.0 : topNY.append((vocabList[i], p1V[i]))
    sortedSF = sorted(topSF, key=lambda pair: pair[1], reverse=True)
    print('SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**')
    for item in sortedSF: print(item[0])
    sortedNY = sorted(topNY, key=lambda pair: pair[1], reverse=True)
    print('NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**')
    for item in sortedNY: print(item[0])

if __name__ == '__main__':

    listOPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    # print(myVocabList)
    # vec1 = setofWords2Vec(myVocabList, listOPosts[0])
    # print(vec1)
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setofWords2Vec(myVocabList, postinDoc))
    p0V, p1V, pAb = trainNB0(trainMat, listClasses)
    # print('p0V = ', p0V)
    # print('p1V = ', p1V)
    # print('pAb = ', pAb)
    #
    # #test
    testingNB(['love', 'my', 'dalmation'])
    testingNB(['stupid', 'garbage'])
    #
    # #SpamTest
    spamTest()

    ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
    # print(ny['entries'])
    sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
    # vocabList, pSF, pNY = localWords(ny, sf)
    getTopWords(ny, sf)

