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
