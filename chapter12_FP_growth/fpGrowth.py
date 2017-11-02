class treeNode:
    ''' FP树节点定义

    '''

    def __init__(self, nameValue, numOccur, parentNode):
        ''' 树节点初始化函数

        :param nameValue: 节点名字
        :param numOccur: 计数值
        :param parentNode: 父节点
        '''
        self.name = nameValue  # 节点名字
        self.count = numOccur  # 计数值
        self.nodeLink = None  # 链接相似的元素项
        self.parent = parentNode  # 父节点
        self.children = {}  # 儿子节点

    def inc(self, numOccur):
        ''' 为当前树节点增加相应的计数值

        :param numOccur: 计数值
        :return:
        '''
        self.count += numOccur

    def disp(self, ind=1):
        ''' 递归遍历FP树，以文本方式显示

        :param ind: 树的深度
        :return:
        '''
        print('  ' * ind, self.name, ' ', self.count)
        for child in self.children.values():
            child.disp(ind + 1)


def loadSimpDat():
    ''' 载入数据集（事务列表）

    :return: 数据集
    '''
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat


def createInitSet(dataSet):
    ''' 将数据集转化为字典形式存储

    :param dataSet: 数据集
    :return: 字典形式的数据集
    '''
    retDict = {}
    # 遍历数据集中的每条事务
    for trans in dataSet:
        # 事务转化为frozenset，作为字典的键
        # 值为事务出现的频率，初始化为1
        retDict[frozenset(trans)] = retDict.get(frozenset(trans), 0) + 1
    return retDict


def updateHeader(nodeToTest, targetNode):
    ''' 更新头指针列表

    :param nodeToTest: 链表头节点
    :param targetNode: 目标节点
    :return:
    '''
    # 不断迭代找到尾节点
    # 举个例子：
    # -----------
    # |  Header |
    # -----------
    # |  {S:7}  |  -> {S:4} -> {S:2} -> None
    # -----------
    # |   ...   |
    # -----------
    # 现在需要把新的节点 targetNode={S:1} 链接到 频繁项集S链表 的尾部
    # 使得链表变成，{S:7} -> {S:4} -> {S:2} -> {S:1} -> None
    while nodeToTest.nodeLink != None:
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode


def updateTree(items, inTree, headerTable, count):
    ''' 将一条新的事务更新进FP树中

    :param items: 事务
    :param inTree: FP树
    :param headerTable: 头指针表
    :param count: 事务出现的频率
    :return: 无
    '''
    # 对该事务中第一个频繁项集进行操作
    curItem = items[0]
    # 判断树的儿子节点中是否包含该频繁项集
    if curItem in inTree.children:
        # 若儿子节点中包含该频繁项集，则直接增加相应的计数值
        inTree.children[curItem].inc(count)
    else:
        # 若儿子节点中不包含该频繁项集，则需要先构造一个节点
        inTree.children[curItem] = treeNode(curItem, count, inTree)
        # 接着判断头指针表中是否存在 「指向该频繁项集的指针」
        if headerTable[curItem][1] == None:
            # 若不存在，则把构造好的节点作为该频繁项集的指针
            headerTable[curItem][1] = inTree.children[curItem]
        else:
            # 若已经存在，则需要把该节点链接到该频繁项集链表的尾部
            updateHeader(headerTable[curItem][1], inTree.children[curItem])
    # 继续递归处理该事务剩余的频繁项集
    if len(items) > 1:
        updateTree(items[1::], inTree.children[curItem], headerTable, count)


def createTree(dataSet, minSup=1):
    ''' 构建FP树

    :param dataSet: 数据集
    :param minSup: 最小支持度
    :return: FP树及头指针表
    '''
    # 初始化头指针表
    headerTable = {}
    # 遍历数据集中的每条事务
    for trans in dataSet:
        # 遍历事务中的每个元素
        for item in trans:
            # 计数
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]
    # 过滤不满足最小支持度的元素
    headerTable = {k : v for k, v in headerTable.items() if v >= minSup}
    # 从头指针表构造初始的频繁项集
    freqItemSet = set(headerTable.keys())
    # 频繁项集为空，直接返回
    if len(freqItemSet) == 0:
        return None, None
    # 对头指针表进行拓展，使得可以存储 「元素的计数值」 及 「指向第一个元素项的指针」
    headerTable = {k : [v, None] for k, v in headerTable.items()}
    # 根节点
    retTree = treeNode('Null Set', 1, None)
    # 遍历数据集中的 每条事务tranSet 及其 出现的频率count
    for tranSet, count in dataSet.items():
        # localD为字典类型，键为当前事务中的频繁项集，值为该频繁项集的全局出现频率，方便对当前事务进行重排序
        localD = {}
        # 遍历当前事务中的每个元素
        for item in tranSet:
            # 判断当前元素是否是频繁项集
            if item in freqItemSet:
                # 若为频繁项集，则从头指针表中取出其全局出现频率
                localD[item] = headerTable[item][0]
        # 判断当前事务的频繁项集是否为空
        if len(localD) > 0:
            # 对当前事务按照频繁项集出现的频率大小进行重排序
            orderedItems = [v[0] for v in sorted(localD.items(), key=lambda p:p[1], reverse=True)]
            # 将处理好的事务插入到FP树中去
            updateTree(orderedItems, retTree, headerTable, count)
    return retTree, headerTable


def ascendTree(leafNode, prefixPath):
    ''' 递归构造给定叶节点的前缀路径

    :param leafNode: 叶节点
    :return:
    '''
    if leafNode.parent != None:
        # 把路径上节点的名字添加进前缀路径中
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)


def findPrefixPath(basePat, treeNode):
    '''构造给定元素的条件模式基

    :param basePat: 元素名字
    :param treeNode: 给定元素的节点
    :return: 条件模式基
    '''
    # 条件模式基
    condPats = {}
    # 遍历给定元素的链表，eg：{S:7} -> {S:4} -> {S:2} -> {S:1} -> None
    while treeNode != None:
        # 获取前缀路径
        prefixPath = []
        ascendTree(treeNode, prefixPath)
        # 构造条件模式基
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink
    return condPats


def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    ''' 递归查找频繁项集

    :param inTree: 当前的FP树
    :param headerTable: FP树对应的头指针表
    :param minSup: 最小支持度
    :param preFix: FP树的前缀
    :param freqItemList: 频繁项集列表
    :return:
    '''
    # 对头指针表的元素按出现频率排序
    bigL = [v[0] for v in sorted(headerTable.items(), key=lambda p: p[1][0])]
    # print('bigL:', bigL)
    # 遍历头指针表中的每个元素(频繁项集)
    for basePat in bigL:
        newFreqSet = preFix.copy()
        # 新的频繁项集，为 preFix + basePat
        newFreqSet.add(basePat)
        # 将新的频繁项集添加进频繁项集列表中
        freqItemList.append(newFreqSet)
        # 构造元素的条件模式基
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        # print('base:', basePat)
        # print('cond:', condPattBases)
        # print('***************************')
        # 根据条件模式基构造条件模式树
        myCondTree, myHead = createTree(condPattBases, minSup)
        # 若头指针表还有元素，继续递归构造
        if myHead != None:
            # print('conditional tree for: ', newFreqSet)
            # myCondTree.disp(1)
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)



if __name__ == '__main__':
    rootNode = treeNode('pyramid', 9, None)
    rootNode.children['eye'] = treeNode('eye', 13, None)
    rootNode.disp()
    rootNode.children['phoenix'] = treeNode('phoenix', 3, None)
    rootNode.disp()

    simpDat = loadSimpDat()
    initSet = createInitSet(simpDat)
    myFPtree, myHeaderTab = createTree(initSet, minSup=3)
    myFPtree.disp()

    print(findPrefixPath('x', myHeaderTab['x'][1]))
    print(findPrefixPath('z', myHeaderTab['z'][1]))
    print(findPrefixPath('r', myHeaderTab['r'][1]))

    freqItems = []
    preFix = set([])
    mineTree(myFPtree, myHeaderTab, 3, preFix, freqItems)
    print(freqItems)