from chapter11_Apriori import apriori


def calcConf(freqSet, H, supportData, brl, minConf=0.7):
    ''' 计算规则可信度，并筛选出满足最小可信度阈值的规则（后件）

    :param freqSet: 频繁项集
    :param H: 规则的后件列表
    :param supportData: 包含支持度数据的字典
    :param brl: 规则列表
    :param minConf: 最小可信度阈值
    :return: 满足阈值的后件列表
    '''
    prunedH = []
    # 遍历每一个后件（consequent）
    for conseq in H:
        # 计算规则的可信度
        # conf(P -> H) = support(P | H) / support(P)
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        # 若满足阈值，则添加进最终的规则列表brl
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH


def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    ''' 不断合并规则后件，生成更多的关联规则。配合书本上图11-4比较好理解

    :param freqSet: 频繁项集
    :param H: 规则的后件列表
    :param supportData: 包含支持度数据的字典
    :param brl: 规则列表
    :param minConf: 最小可信度阈值
    :return: 无
    '''
    # 规则后件的大小m
    m = len(H[0])
    # print(m)
    # 由于接下来规则后件的大小从m合并成m+1
    # 所以需要判断当前频繁项集是否可以移除大小为m+1的后件
    if (len(freqSet) > (m + 1)):
        # 调用aprioriGen，使得规则后件大小从m合并成m+1
        Hmp1 = apriori.aprioriGen(H, m + 1)
        # 筛选出满足最小可信度阈值的后件
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf)
        # 如果后件列表元素大于1个，则可以继续合并
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)


def generateRules(L, supportData, minConf=0.7):
    ''' 从频繁项集从生成规则列表

    :param L: 频繁项集列表
    :param supportData: 包含频繁项集支持数据的字典
    :param minConf: 最小可信度阈值
    :return: 规则列表
    '''
    bigRuleList = []
    # 从L2开始，因为无法从单元素项集中构建规则
    for i in range(1, len(L)):
        # 遍历每一个频繁项集
        for freqSet in L[i]:
            # 遍历频繁项集中的每一个元素，构造规则的后件列表
            # eg. freqSet = {0, 1, 2}, 则H1 = [{0}, {1}, {2}]
            H1 = [frozenset([item]) for item in freqSet]
            if i > 1: # 频繁项元素大于2个的情况
                # 规则右件大小为1的情况，形如 {A, B, ...} -> {C}
                H1 = calcConf(freqSet, H1, supportData, bigRuleList, minConf)
                # 不断合并规则后件，生成更多的关联规则
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: # 频繁项元素只有2个的情况，直接生成规则，形如 {A} -> {B}
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList


if __name__ == '__main__':
    dataSet = apriori.loadDataSet()

    L, suppData = apriori.apriori(dataSet, minSupport=0.5)

    rules = generateRules(L, suppData, minConf=0.7)
    print(rules)

    rules = generateRules(L, suppData, minConf=0.5)
    print(rules)
