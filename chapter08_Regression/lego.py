import numpy as np
from bs4 import BeautifulSoup
from chapter08_Regression import regression
from chapter08_Regression import ridgeRegression


# 由于googleapis不可用，故从页面读取数据，生成retX和retY列表
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    # 打开并读取HTML文件
    fr = open(inFile, 'rb')
    soup = BeautifulSoup(fr.read(), 'lxml')
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        currentRow = soup.findAll('table', r="%d" % i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print("item #%d did not sell" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if  sellingPrice > origPrc * 0.5:
                    print("%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice))
                    retX.append([yr, numPce, newFlag, origPrc])
                    retY.append(sellingPrice)
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)


# 依次读取六种乐高套装的数据，并生成数据矩阵
def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    # 岭回归中使用了30个不同的λ创建了不同的回归系数
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        np.random.shuffle(indexList)
        for j in range(m):
            if j < int(0.9 * m):
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        wMat = ridgeRegression.ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = np.mat(testX)
            matTrainX = np.mat(trainX)
            meanTrain = np.mean(matTrainX, 0)
            varTrain = np.var(matTrainX, 0)
            matTestX = (matTestX - meanTrain) / varTrain
            yEst = matTestX * np.mat(wMat[k, :]).T + np.mean(trainY)
            errorMat[i, k] = regression.rssError(yEst.T.A, np.array(testY))
    meanErrors = np.mean(errorMat, 0)
    minMean = float(np.min(meanErrors))
    # print('minMean', minMean)
    # np.nonzero(A) 返回A中非零元素的索引
    bestWeights = wMat[np.nonzero(meanErrors == minMean)]
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    meanX = np.mean(xMat, 0)
    varX = np.var(xMat, 0)
    unreg = bestWeights / varX
    print('the best model from Ridge Regression is:\n', unreg)
    print('with constant term: ', -1 * np.sum(np.multiply(meanX, unreg)) + np.mean(yMat))


if __name__ =='__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    m, n = np.shape(lgX)
    lgX1 = np.mat(np.ones((m, 5)))
    lgX1[:, 1:5] = np.mat(lgX)
    # print(lgX[0])
    # print(lgX1[0])
    ws = regression.standRegress(lgX1, lgY)
    # print(ws)
    # print(lgX1[0] * ws)
    # print(lgX1[-1] * ws)
    # print(lgX1[43] * ws)
    crossValidation(lgX, lgY, 10)
