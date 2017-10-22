import numpy as np


def printMat(inMat, thresh=0.8):
    ''' 根据阈值打印01矩阵

    :param inMat: 打印矩阵
    :param thresh: 控制输出阈值
    :return:
    '''
    for i in range(32):
        for k in range(32):
            if float(inMat[i, k]) > thresh:
                print(1, end='')
            else:
                print(0, end='')
        print('')


def imgCompress(numSV=3, thresh=0.8):
    ''' 使用numSV个奇异值进行SVD图像压缩

    :param numSV: 奇异值个数
    :param thresh: 阈值
    :return:
    '''
    myl = []
    for line in open('0_5.txt').readlines():
        newRow = []
        for i in range(32):
            newRow.append(int(line[i]))
        myl.append(newRow)
    myMat = np.mat(myl)
    print('******original matrix******')
    printMat(myMat, thresh)
    U, Sigma, VT = np.linalg.svd(myMat)
    SigRecon = np.mat(np.zeros((numSV, numSV)))
    for k in range(numSV): SigRecon[k, k] = Sigma[k]
    reconMat = U[:, :numSV] * SigRecon * VT[:numSV, :]
    print('******reconstructed matrix using %d singular values******' % numSV)
    printMat(reconMat, thresh)


if __name__ == '__main__':
    imgCompress(2)