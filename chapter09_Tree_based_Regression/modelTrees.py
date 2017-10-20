import numpy as np
from chapter09_Tree_based_Regression import regTrees


def linearSolve(dataSet):
    ''' 求解线性回归系数

    :param dataSet: 数据集
    :return: 回归系数，格式化后的样本及标签
    '''
    m, n = np.shape(dataSet) # 数据集大小及特征数目
    X = np.mat(np.ones((m, n)))
    Y = np.mat(np.ones((m, 1)))
    # 数据集前n-1列为特征
    X[:, 1:n] = dataSet[:, 0:n-1]
    # 数据集最后一列为标签
    Y = dataSet[:, -1]
    xTx = X.T * X
    # 判断矩阵是否可逆
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse,\n\
                        try increasing the second value of ops')
    # 用正规方程求解系数
    ws = xTx.I * X.T * Y
    return ws, X, Y


def modelLeaf(dataSet):
    ''' 叶节点为线性回归模型

    :param dataSet: 数据集
    :return: 拟合数据集的线性回归系数
    '''
    ws, X, Y = linearSolve(dataSet)
    return ws


def modelErr(dataSet):
    ''' 误差计算函数

    :param dataSet: 数据集
    :return: 误差
    '''
    ws, X, Y = linearSolve(dataSet)
    yHat = X * ws
    # 误差为真实值与预测值之间的误差平方和
    return np.sum(np.power(yHat - Y, 2))


if __name__ == '__main__':
    myMat2 = np.mat(regTrees.loadDataSet('exp2.txt'))
    # draw dataSet
    regTrees.draw(myMat2[:, 0].tolist(), myMat2[:, 1].tolist())
    myTree2 = regTrees.createTree(myMat2, leafType=modelLeaf, errType=modelErr, ops=(1, 10))
    print(myTree2)
