import numpy as np


def loadExData():
    ''' 载入数据

    :return:
    '''
    return[[1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1]]


def loadExData2():
    ''' 载入数据

    :return:
    '''
    return[[4, 4, 0, 2, 2],
           [4, 0, 0, 3, 3],
           [4, 0, 0, 1, 1],
           [1, 1, 1, 2, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]


def loadExData3():
    return[[2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]


def ecludSim(inA, inB):
    ''' 欧几里得相似度

    :param inA: 样本A
    :param inB: 样本B
    :return: 欧几里得相似度
    '''
    return 1.0 / (1.0 + np.linalg.norm(inA - inB))


def pearsSim(inA, inB):
    ''' 皮尔逊相关系数

    :param inA: 样本A
    :param inB: 样本B
    :return: 皮尔逊相关系数
    '''
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=False)[0][1]


def cosSim(inA, inB):
    ''' 余弦相似度

    :param inA: 样本A
    :param inB: 样本B
    :return: 余弦相似度
    '''
    num = float(inA.T * inB)
    denom = np.linalg.norm(inA) * np.linalg.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    ''' 计算用户对物品的评分

    :param dataMat: 数据集
    :param user: 用户
    :param simMeas: 相似性度量函数
    :param item: 物品
    :return: 评分
    '''
    n = np.shape(dataMat)[1]  # 物品个数
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):  # 遍历每个物品
        userRating = dataMat[user, j]  # 获取用户对第j个物品的评分
        if userRating == 0: continue  # 未评分就跳过
        # 获取那些对第item个物品和第j个物品都评过分的用户编号
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        if len(overLap) == 0:  # 不存在重合元素，则相似度为0
            similarity = 0
        else:  # 计算第j个物品和第item个物品的相似度
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        # print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity  # 累加相似度
        ratSimTotal += similarity * userRating  # 加权相似度
    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal


def sigmaPct(sigma, percentage=0.9):
    ''' 按照前k个奇异值的平方和占总奇异值的平方和的百分比来确定k的值

    :param sigma: 奇异值列表
    :param percentage: 损失的阈值
    :return: 确定后的k值
    '''
    sg2 = sigma**2  # 对sigma求平方
    sumsg2 = np.sum(sg2)  # 求所有奇异值sigma的平方和
    sumk = 0  # sumk是前k个奇异值的平方和
    k = 0
    for i in sigma:
        sumk += i**2
        k += 1
        if sumk >= sumsg2 * percentage:
            return k


def svdEst(dataMat, user, simMeas, item):
    ''' 使用SVD计算用户对物品的评分

    :param dataMat: 数据集
    :param user: 用户
    :param simMeas: 相似性度量函数
    :param item: 物品
    :return: 评分
    '''
    n = np.shape(dataMat)[1]  # 物品个数
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = np.linalg.svd(dataMat)  # 奇异值分解
    k = sigmaPct(Sigma)
    Sigk = np.mat(np.eye(k) * Sigma[:k])  # 构造对角阵
    # 根据k的值将原始数据转换到低维空间,xformedItems表示物品在k维空间转换后的值
    xformedItems = dataMat.T * U[:, :k] * Sigk.I
    for j in range(n):  # 遍历每个物品
        userRating = dataMat[user, j]  # 获取用户对第j个物品的评分
        # 若该物品未评分 或 该物品就是要评分的物品，那么就跳过
        if userRating == 0 or j == item: continue
        # 计算相似度
        similarity = simMeas(xformedItems[item, :].T, xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    ''' 为用户推荐评分最高的N个物品

    :param dataMat: 数据集
    :param user: 用户
    :param N: 推荐物品的个数
    :param simMeas: 相似性度量函数
    :param estMethod: 评分计算方式
    :return: topN物品
    '''
    # 用户未评分的物品编号
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    # 未评分物品的个数
    num = len(unratedItems)
    # 所有物品都评过分
    if num == 0:
        return 'you rated everything'
    # N与num取个min
    if N > num:
        N = num
    itemScores = []
    # 遍历每个未评分的物品
    for item in unratedItems:
        # 计算用户对第item个物品的评分
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((len(itemScores) + 1, estimatedScore))
    # 排序后取评分最高的前N个物品
    return sorted(itemScores, key=lambda k: k[1], reverse=True)[:N]


if __name__ == '__main__':
    U, sigma, VT = np.linalg.svd([[1, 1], [7, 7]])
    print(U)
    print(sigma)
    print(VT)

    Data = loadExData()
    U, sigma, VT = np.linalg.svd(Data)
    print(sigma)
    Sig3 = np.mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[2]]])
    print(Sig3)
    mat = U[:, :3] * Sig3 * VT[:3, :]
    print(mat)

    myMat = np.mat(loadExData())
    # 欧几里得
    print(ecludSim(myMat[:, 0], myMat[:, 4]))
    print(ecludSim(myMat[:, 0], myMat[:, 0]))
    # 余弦相似
    print(cosSim(myMat[:, 0], myMat[:, 4]))
    print(cosSim(myMat[:, 0], myMat[:, 0]))
    # 皮尔逊系数
    print(pearsSim(myMat[:, 0], myMat[:, 4]))
    print(pearsSim(myMat[:, 0], myMat[:, 0]))

    myMat = np.mat(loadExData2())
    print(recommend(myMat, 2))
    print(recommend(myMat, 2, simMeas=ecludSim))
    print(recommend(myMat, 2, simMeas=pearsSim))

    myMat = np.mat(loadExData3())
    U, Sigma, VT = np.linalg.svd(myMat)
    print(Sigma)
    Sig = Sigma**2
    print(sum(Sig))  # 553.0
    print(sum(Sig) * 0.9)  # 497.7
    print(sum(Sig[:2]))  # 388.794482903
    print(sum(Sig[:3]))  # 500.522599693

    print(recommend(myMat, 1, estMethod=svdEst))
