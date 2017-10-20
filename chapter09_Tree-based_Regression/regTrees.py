import numpy as np
import matplotlib.pyplot as plt


def loadDataSet(fileName):
    ''' 载入文件

    :param fileName: 文件名
    :return: 数据矩阵
    '''
    dataMat = []
    fr = open(fileName)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        # 将每行映射成浮点数
        # 另外，python3中map返回一个iterator，需要转成list才行
        fltLine = list(map(float, curLine))
        dataMat.append(fltLine)
    return dataMat


def binSplitDataSet(dataSet, feature, value):
    ''' 将数据集按特征取值切分成两个子集

    :param dataSet: 数据集
    :param feature: 待切分的特征
    :param value: 该特征的某个取值
    :return: 切分后的两个子集
    '''
    # mat0: 数据集中第feature个特征的取值大于value的样本
    mat0 = dataSet[np.nonzero(dataSet[:, feature] > value)[0], :]
    # mat0: 数据集中第feature个特征的取值小等于value的样本
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <= value)[0], :]
    return mat0, mat1


def regLeaf(dataSet):
    ''' 生成叶节点

    :param dataSet: 数据集
    :return: 标签的均值
    '''
    return np.mean(dataSet[:, -1])


def regErr(dataSet):
    ''' 误差计算函数，衡量一个数据集的混乱程度

    :param dataSet: 数据集
    :return: 标签的总方差
    '''
    # 总方差 = 均方差 * 样本数
    return np.var(dataSet[:, -1]) * np.shape(dataSet)[0]


def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    ''' 找到最佳的二元分割方式

    :param dataSet: 数据集
    :param leafType: 叶节点类型
    :param errType: 误差计算函数类型
    :param ops: 构建树所需的其它可选参数
    :return: 最优切分特征及其取值
    '''
    tolS = ops[0]  # 容许的误差下降值
    tolN = ops[1]  # 切分的最少样本数
    # 数据集中所有标签都相等，则退出
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    m, n = np.shape(dataSet)  # m: 样本数 n: 特征数
    S = errType(dataSet)  # 数据集上的总误差
    bestS = np.inf  # 初始化
    bestIndex = 0  # 最佳特征的索引
    bestValue = 0  # 最佳特征的取值
    for featIndex in range(n - 1):  # 　遍历所有的特征
        for splitVal in set(
                (dataSet[:, featIndex].T.tolist())[0]):  # 遍历该特征所有可能的取值
            # 按当前的特征和取值将数据集切分成两个子集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 若切分后的子集样本数小于定义的最小样本数totN，则忽略该切分方式
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
                continue
            # 计算该切分方式的误差
            newS = errType(mat0) + errType(mat1)
            # 更新最优值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 若减少的误差值小于容许的误差下降值（即切分效果不大），则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 若切分后的子集样本数小于定义的最小样本数totN，则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最优切分特征及其取值
    return bestIndex, bestValue


def createTree(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    ''' 构建树

    :param dataSet: 数据集
    :param leafType: 叶节点类型
    :param errType: 误差计算函数类型
    :param ops: 构建树所需的其它可选参数
    :return: 构建好的树
    '''
    # 找到最佳的二元分割方式
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)
    # 满足条件时，返回叶节点
    if feat is None:
        return val
    retTree = {}
    retTree['spInd'] = feat  # 分割特征
    retTree['spVal'] = val  # 分割取值
    # 分割后的两个子集
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    # 递归构造左右子树
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


def isTree(obj):
    ''' 判断对象obj是否为一棵树（树用字典存储）

    :param obj: 对象object
    :return: 判断obj是否为字典类型
    '''
    return type(obj).__name__ == 'dict'


def draw(x, y):
    ''' 可视化2-D数据集

    :param x: 横坐标
    :param y: 纵坐标
    :return:
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x, y)
    plt.show()


def getMean(tree):
    ''' 对树进行塌陷处理（即返回树的平均值）

    :param tree: 树节点
    :return: 树的平均值
    '''
    if isTree(tree['right']):
        tree['right'] = getMean(tree['right'])
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def prune(tree, testData):
    ''' 对树进行剪枝

    :param tree: 树节点
    :param testData: 测试数据集
    :return: 剪枝后的树
    '''
    # 测试数据集为空，则直接剪枝
    if np.shape(testData)[0] == 0:
        return getMean(tree)
    # 对非叶节点进行分割
    if isTree(tree['right']) or isTree(tree['left']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    # 递归修剪左子树
    if isTree(tree['left']):
        tree['left'] = prune(tree['left'], lSet)
    # 递归修剪右子树
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'], rSet)
    # 若修剪后两个分支已不再是子树，则判断是否进行合并
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 不合并的误差
        errorNoMerge = np.sum(np.power(
            lSet[:, -1] - tree['left'], 2)) + np.sum(np.power(rSet[:, -1] - tree['right'], 2))
        treeMean = (tree['left'] + tree['right']) / 2.0
        # 合并后的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 合并后的误差更小，则合并
        if errorMerge < errorNoMerge:
            print('merging')
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    fileNameList = ['ex00.txt', 'ex0.txt', 'ex2.txt']
    for fileName in fileNameList:
        xc = 0
        yc = 1
        myDat = loadDataSet(fileName)
        myMat = np.mat(myDat)
        if fileName == 'ex0.txt':
            xc += 1
            yc += 1
        draw(myMat[:, xc].tolist(), myMat[:, yc].tolist())
        myTree = createTree(myMat)
        print(myTree)

    myDat2 = loadDataSet('ex2.txt')
    myMat2 = np.mat(myDat2)
    myTree = createTree(myMat2, ops=(0, 1))
    myDat2Test = loadDataSet('ex2test.txt')
    myMat2Test = np.mat(myDat2Test)
    prunedTree = prune(myTree, myMat2Test)
    print(prunedTree)
