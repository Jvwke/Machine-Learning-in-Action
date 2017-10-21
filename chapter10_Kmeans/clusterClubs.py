import numpy as np
import matplotlib.pyplot as plt
from chapter10_Kmeans import kmeans


def distSLC(vecA, vecB):
    ''' 根据经纬度计算地球表面任意两点的距离

    :param vecA: 地点A
    :param vecB: 地点B
    :return: A、B之间的距离
    '''
    a = np.sin(vecA[0, 1] * np.pi / 180.0) * np.sin(vecB[0, 1] * np.pi / 180.0)
    b = np.cos(vecA[0, 1] * np.pi / 180.0) * np.cos(vecB[0, 1] * np.pi / 180.0) * \
        np.cos(np.pi * (vecB[0, 0] - vecA[0, 0]) / 180.0)
    return np.arccos(a + b) * 6371.0


def clusterClubs(numClust=5):
    ''' 将俱乐部进行聚类并画出结果

    :param numClust: 簇的个数
    :return:
    '''
    datList = []
    for line in open('places.txt').readlines():
        lineArr = line.split('\t')
        datList.append([float(lineArr[4]), float(lineArr[3])]) # 纬度、经度
    datMat = np.mat(datList)
    # 使用二分-k均值聚类
    myCentroids, clustAgging = kmeans.biKmeans(datMat, numClust, distMeas=distSLC)
    fig = plt.figure()
    rect = [0.1, 0.1, 0.8, 0.8]
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    axprops = dict(xticks=[], yticks=[])
    ax0 = fig.add_axes(rect, label='ax0', **axprops)
    imgP = plt.imread('Portland.png')
    ax0.imshow(imgP)
    ax1 = fig.add_axes(rect, label='ax1', frameon=False)
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAgging[:, 0].A == i)[0], :]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        ax1.scatter(ptsInCurrCluster[:, 0].flatten().A[0],
                    ptsInCurrCluster[:, 1].flatten().A[0],
                    marker=markerStyle,
                    s=90)
    ax1.scatter(myCentroids[:, 0, 0],
                myCentroids[:, 0, 1],
                marker='+',
                s=300)
    plt.show()


if __name__ == '__main__':
    clusterClubs()