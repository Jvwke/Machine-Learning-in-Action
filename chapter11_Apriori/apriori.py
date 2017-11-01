def loadDataSet():
    ''' 载入数据集

    :return: 数据集
    '''
    return [
        [1, 3, 4],
        [2, 3, 5],
        [1, 2, 3, 5],
        [2, 5]
    ]


def createC1(dataSet):
    ''' 计算C1项集

    :param dataSet: 数据集
    :return: C1项集
    '''
    C1 = []
    # 遍历每条交易记录
    for transaction in dataSet:
        # 遍历交易记录中的每件商品
        for item in transaction:
            # 如果不在C1项目集中，则添加
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return list(map(frozenset, C1))


def scanD(D, Ck, minSupport):
    ''' 从候选项集Ck生成频繁项集Lk, Ck -> Lk

    :param D: 数据集
    :param Ck: 候选项集列表
    :param minSupport: 最小支持度
    :return: 频繁项集列表及对应的支持度
    '''
    ssCnt = {}
    # 遍历每条交易记录
    for tid in D:
        # 遍历每个候选集
        for can in Ck:
            # 如果该候选集是该条交易记录的一个子集
            if can.issubset(tid):
                # 计数
                if not (can in ssCnt):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    # 交易记录总数
    numItems = float(len(D))
    # 满足最小支持度的频繁项集列表及其支持度
    retList = []
    supportData = {}
    for key in ssCnt:
        # 计算支持度
        support = ssCnt[key] / numItems
        # 若大于最小支持度，则添加进频繁项集列表
        if support >= minSupport:
            retList.insert(0, key)
        # 记录支持度
        supportData[key] = support
    return retList, supportData


def aprioriGen(Lk_1, k):
    ''' 从频繁项集Lk-1生成候选项集Ck, Lk-1 -> Ck

    :param Lk: 频繁项集Lk-1
    :param k: 项集元素个数
    :return: 候选项集Ck
    '''
    # 候选项集列表
    retList = []
    # 频繁项集Lk-1的个数
    lenLk_1 = len(Lk_1)
    # 遍历Lk的各种组合
    for i in range(lenLk_1):
        for j in range(i + 1, lenLk_1):
            # 注意到Lk_1[i]和Lk_1[j]的小大为k-1
            # 取Lk_1[i]和Lk_1[j]的前k-2项，刚好还剩下一项
            L1 = list(Lk_1[i])[:k - 2]
            L2 = list(Lk_1[j])[:k - 2]
            # 排序之后方便比较
            L1.sort()
            L2.sort()
            # 如果Lk_1[i]和Lk_1[j]的前k-2项相等，则可以合并形成新的候选集
            # 该新的候选集的大小为 k-2 + 1 + 1 = k
            if L1 == L2:
                retList.append(Lk_1[i] | Lk_1[j])
    return retList


def apriori(dataSet, minSupport=0.5):
    ''' Apriori算法

    :param dataSet: 数据集
    :param minSupport: 最小支持度
    :return: 各阶频繁项集及其支持度
    '''
    # 创建C1
    C1 = createC1(dataSet)
    # 将数据集映射成集合列表
    D = list(map(set, dataSet))
    print(D)
    # 创建L1
    L1, supportData = scanD(D, C1, minSupport)
    # 频繁项集列表，初始只有L1
    L = [L1]
    k = 2
    # 不断循环，直到无法生成更多的项集
    while (len(L[k - 2]) > 0):
        # 从频繁项集Lk-1生成候选项集Ck
        Ck = aprioriGen(L[k - 2], k)
        # 从候选项集Ck生成频繁项集Lk
        Lk, supK = scanD(D, Ck, minSupport)
        # 更新支持度
        supportData.update(supK)
        # 更新频繁项集列表
        L.append(Lk)
        k += 1
    return L, supportData


if __name__ == '__main__':
    dataSet = loadDataSet()

    C1 = createC1(dataSet)
    print(C1)
    L1, supportData0 = scanD(dataSet, C1, 0.5)
    print(L1)

    L, supportData = apriori(dataSet)
    print(L)
    print(supportData)
