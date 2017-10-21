import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName, delim='\t'):
    ''' 载入数据集

    :param fileName: 文件名
    :param delim: 分隔符
    :return: 数据矩阵
    '''
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float, line)) for line in stringArr]
    return np.mat(datArr)


def pca(dataMat, topNFeat=9999999):
    ''' 主成分分析

    :param dataMat: 数据集
    :param topNFeat: 保留topN特征
    :return: 降维后的数据，恢复后的数据
    '''
    # 每个特征的平均值
    meanVals = np.mean(dataMat, axis=0)
    # 减掉平均值
    meanRemoved = dataMat - meanVals
    # 协方差矩阵
    covMat = np.cov(meanRemoved, rowvar=False)
    # 特征值和特征向量
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    # 对特征值排序
    eigValInd = np.argsort(eigVals)
    # 取特征值最大的N个
    eigValInd = eigValInd[:-(topNFeat + 1):-1]
    redEigVects = eigVects[:, eigValInd]
    # 将原始数据映射到低维空间
    lowDDataMat = meanRemoved * redEigVects
    # 将低维数据恢复到高维空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    return lowDDataMat, reconMat


def replaceNanWithMean():
    ''' 将Nan替换为对应特征的平均值

    :return:
    '''
    dataMat = loadDataSet('secom.data', ' ')
    numFeat = np.shape(dataMat)[1]
    for i in range(numFeat):
        meanVal = np.mean(dataMat[np.nonzero(~np.isnan(dataMat[:, i].A))[0], i])
        dataMat[np.nonzero(np.isnan(dataMat[:, i].A))[0], i] = meanVal
    return dataMat


if __name__ == '__main__':
    dataMat = loadDataSet('testSet.txt')
    lowDMat, reconMat = pca(dataMat, 1)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

    dataMat = replaceNanWithMean()
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVects = np.linalg.eig(np.mat(covMat))
    print(eigVals)