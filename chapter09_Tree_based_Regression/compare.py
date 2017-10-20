import numpy as np
from chapter09_Tree_based_Regression import regTrees
from chapter09_Tree_based_Regression import modelTrees


def regTreeEval(model, inDat):
    ''' 回归树预测函数

    :param model: 模型参数
    :param inDat: 待预测数据
    :return: 预测结果
    '''
    return float(model)


def modelTreeEval(model, inDat):
    ''' 模型树预测函数

    :param model: 模型参数
    :param inDat: 待预测数据
    :return: 预测结果
    '''
    m, n = np.shape(inDat)
    X = np.mat(np.ones((1, n + 1)))
    X[:, 1:n + 1] = inDat  # 格式化预测数据
    return float(X * model)


def treeForeCast(tree, inData, modelEval=regTreeEval):
    ''' 树预测

    :param tree: 树
    :param inData: 输入数据
    :param modelEval: 树的类型
    :return: 预测结果
    '''
    # 不是树类型
    if not regTrees.isTree(tree):
        return modelEval(tree, inData)
    # 二元划分，进入左子树
    if inData[:, tree['spInd']] > tree['spVal']:
        if regTrees.isTree(tree['left']):  # 左子树仍然是棵树，递归预测
            return treeForeCast(tree['left'], inData, modelEval)
        else:  # 到达叶节点，直接预测
            return modelEval(tree['left'], inData)
    else:  # 进入右子树
        if regTrees.isTree(tree['right']):  # 右子树仍然是棵树，递归预测
            return treeForeCast(tree['right'], inData, modelEval)
        else:  # 到达叶节点，直接预测
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=regTreeEval):
    ''' 对测试集进行预测

    :param tree: 构建好的树
    :param testData: 测试数据集
    :param modelEval: 树的类型
    :return: 预测值
    '''
    m = len(testData)  # 测试集大小
    yHat = np.mat(np.zeros((m, 1)))  # 预测值矩阵
    for i in range(m):  # 对每个测试样本进行预测
        yHat[i, 0] = treeForeCast(tree, np.mat(testData[i]), modelEval)
    return yHat


if __name__ == '__main__':
    trainMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_train.txt'))
    # regTrees.draw(trainMat[:, 0].tolist(), trainMat[:, 1].tolist())
    testMat = np.mat(regTrees.loadDataSet('bikeSpeedVsIq_test.txt'))
    # regTrees.draw(testMat[:, 0].tolist(), testMat[:, 1].tolist())

    myTree = regTrees.createTree(trainMat, ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0])
    # print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    # 0.964085231822

    myTree = regTrees.createTree(
        trainMat,
        leafType=modelTrees.modelLeaf,
        errType=modelTrees.modelErr,
        ops=(1, 20))
    yHat = createForeCast(myTree, testMat[:, 0], modelEval=modelTreeEval)
    # print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    # 0.976041219138

    ws, X, Y = modelTrees.linearSolve(trainMat)
    for i in range(np.shape(testMat)[0]):
        yHat[i] = testMat[i, 0] * ws[1, 0] + ws[0, 0]
    # print(ws)
    # ws = [[37.58916794]
    #       [6.18978355]]
    # print(np.corrcoef(yHat, testMat[:, 1], rowvar=0)[0, 1])
    # 0.943468423567
