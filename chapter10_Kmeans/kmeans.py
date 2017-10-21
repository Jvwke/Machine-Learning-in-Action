import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    ''' 载入数据集

    :param fileName: 文件名
    :return: 数据矩阵
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def distEclud(vecA, vecB):
    ''' 计算两个样本之间的欧几里得距离

    :param vecA: 样本A
    :param vecB: 样本B
    :return: A与B之间的欧几里得距离
    '''
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))


def randCent(dataSet, k):
    ''' 在数据集中随机产生k个簇心

    :param dataSet: 数据集
    :param k: 簇心个数
    :return: 随机生成的k个簇心
    '''
    m, n = np.shape(dataSet)  # 数据集大小
    centroids = np.mat(np.zeros((k, n)))  # 初始化
    for j in range(n):  # 遍历每个特征
        minJ = np.min(dataSet[:, j])  # 第j个特征的最小值
        maxJ = np.max(dataSet[:, j])  # 第j个特征的最大值
        rangeJ = float(maxJ - minJ)  # 第j个特征的取值范围
        # 使用np.random.rand函数生成每个簇心第j个特征的取值
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    ''' k-均值

    :param dataSet: 数据集
    :param k: 簇心个数
    :param distMeas: 距离度量函数
    :param createCent: 簇心初始化函数
    :return: k个簇心及每个样本的分配情况
    '''
    m, n = np.shape(dataSet)  # 样本大小
    # 样本分配情况，第一列记录簇索引值，第二列记录误差大小
    clusterAssment = np.mat(np.zeros((m, 2)))
    # 生成k个簇心
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        # 遍历所有样本
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            # 遍历所有簇心
            for j in range(k):
                # 计算第i个样本到第j个簇心之间的距离
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                # 更新最小值
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            # 跟上一轮比，判断簇心是否改变
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, 0] = minIndex
        print(centroids)
        # 更新簇心
        for cent in range(k):
            # 找到样本中属于第cent个簇心的样本，用来更新簇心
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def biKmeans(dataSet, k, distMeas=distEclud):
    ''' 二分k-均值算法

    :param dataSet: 数据集
    :param k: 簇个数
    :param distMeas: 距离度量函数
    :return: k个簇心及每个样本的分配情况
    '''
    m, n = np.shape(dataSet)  # 数据集大小
    clusterAssment = np.mat(np.zeros((m, 2)))  # 分配初始化
    centroid0 = np.mean(dataSet, axis=0).tolist()[0]  # 初始簇心
    centList = [centroid0]  # 簇心列表
    for j in range(m):  # 计算样本与初始簇心之间的误差
        clusterAssment[j, 1] = distEclud(dataSet[j, :], centroid0)**2
    while len(centList) < k:  # 尚未生成k个簇
        lowestSSE = np.inf  # 最小的平方误差和
        bestCentToSplit = -1  # 最优分割簇的索引
        for i in range(len(centList)):  # 遍历每个簇
            # 分离出属于第i个簇的样本
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == i)[0], :]
            num = np.shape(ptsInCurrCluster)[0]
            # 若样本数少于2个，不必再二分
            if num < 2: continue
            # 将第i个簇二分
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            # 第i个簇分割后的sse
            sseSplit = np.sum(splitClustAss[:, 1])
            # 第i个簇之外(那些未被分割的簇)的sse
            sseNotSplit = np.sum(
                clusterAssment[np.nonzero(clusterAssment[:, 0].A != i)[0], 1])
            # 总的sse: 第i个簇分割后的sse + 第i个簇之外那些簇的sse
            if (sseSplit + sseNotSplit) < lowestSSE:  # 更新最优分割的簇
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        # 更新二分后的簇编号，一类是原始的编号bestCentToSplit，另一类是新的编号len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 1)[0], 0] = len(centList)
        bestClustAss[np.nonzero(bestClustAss[:, 0].A == 0)[0], 0] = bestCentToSplit
        print('the bestCentToSplit is: ', bestCentToSplit)
        print('the len of bestClustAss is: ', len(bestClustAss))
        # 更新二分后两个簇心的位置
        centList[bestCentToSplit] = bestNewCents[0, :] # 改变原来第bestCentToSplit个簇心位置
        centList.append(bestNewCents[1, :])  # 新增一个簇
        # 更新匹配情况
        clusterAssment[np.nonzero(clusterAssment[:, 0].A == bestCentToSplit)[0], :] = bestClustAss
    return np.array(centList), clusterAssment


if __name__ == '__main__':
    datMat = np.mat(loadDataSet('testSet.txt'))
    myCentroids, clusterAssing = kMeans(datMat, k=4)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 绘制数据集
    ax.scatter(datMat[:, 0].tolist(), datMat[:, 1].tolist())
    # 绘制簇心
    ax.scatter(myCentroids[:, 0].tolist(), myCentroids[:, 1].tolist(), c='r', marker='x')
    plt.show()

    datMat2 = np.mat(loadDataSet('testSet2.txt'))
    centList, myNewAssments = biKmeans(datMat2, k=3)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datMat2[:, 0].tolist(), datMat2[:, 1].tolist())
    ax.scatter(centList[:, 0, 0], centList[:, 0, 1], c='r', marker='x')
    plt.show()
