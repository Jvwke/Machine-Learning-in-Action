import numpy as np
from tkinter import *
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from chapter09_Tree_based_Regression import regTrees
from chapter09_Tree_based_Regression import modelTrees
from chapter09_Tree_based_Regression import compare


def reDraw(tolS, tolN):
    ''' 根据参数tolS和tolN来重绘

    :param tolS: 容许的误差下降值
    :param tolN: 切分的最少样本数
    :return:
    '''
    reDraw.f.clf()  # 清空figure
    reDraw.a = reDraw.f.add_subplot(111) # 添加新的子图
    if chkBtnVar.get(): # 选中复选框，构建模型树
        if tolN < 2: tolN = 2 # tolN最小为2
        myTree = regTrees.createTree(reDraw.rawDat, modelTrees.modelLeaf, modelTrees.modelErr, (tolS, tolN))
        yHat = compare.createForeCast(myTree, reDraw.testDat, compare.modelTreeEval)
    else: # 构建回归树
        myTree = regTrees.createTree(reDraw.rawDat, ops=(tolS, tolN))
        yHat = compare.createForeCast(myTree, reDraw.testDat)
    # 使用散点图绘制真实值
    reDraw.a.scatter(reDraw.rawDat[:, 0].tolist(), reDraw.rawDat[:, 1].tolist(), s=5)
    # 使用plot绘制预测值
    reDraw.a.plot(reDraw.testDat, yHat, linewidth=2.0)
    reDraw.canvas.show()


def getInputs():
    ''' 从输入框中提取参数tolN和tolS

    :return:
    '''
    try:
        tolN = int(tolNentry.get())  # 得到输入框中的tolN
    except BaseException:  # 处理异常
        tolN = 10
        print('enter Integer for tolN')
        tolNentry.delete(0, END)
        tolNentry.insert(0, '10')  # 恢复默认值
    try:
        tolS = float(tolSentry.get()) # 得到输入框中的tolS
    except BaseException: # 处理异常
        tolS = 1.0
        print('enter Float for tolS')
        tolSentry.delete(0, END)
        tolSentry.insert(0, '1.0') # 恢复默认值
    return tolN, tolS


def drawNewTree():
    ''' 绘制新的回归/模型树

    :return:
    '''
    tolN, tolS = getInputs()  # get values from Entry boxes
    reDraw(tolS, tolN)


if __name__ == '__main__':
    # 添加tkinter窗口
    root = Tk()
    # 添加画布
    reDraw.f = Figure(figsize=(5, 4), dpi=100)
    reDraw.canvas = FigureCanvasTkAgg(reDraw.f, master=root)
    reDraw.canvas.show()
    reDraw.canvas.get_tk_widget().grid(row=0, columnspan=3)
    # 添加tolN标签及输入框
    Label(root, text='tolN').grid(row=1, column=0)
    tolNentry = Entry(root)
    tolNentry.grid(row=1, column=1)
    tolNentry.insert(0, '10')
    # 添加tolS标签及输入框
    Label(root, text='tolS').grid(row=2, column=0)
    tolSentry = Entry(root)
    tolSentry.grid(row=2, column=1)
    tolSentry.insert(0, '1.0')
    # 添加重绘按钮
    Button(root, text='ReDraw', command=drawNewTree).grid(row=1, column=2, rowspan=3)
    # 添加复选框
    chkBtnVar = IntVar()
    chkBtn = Checkbutton(root, text='Model Tree', variable=chkBtnVar)
    chkBtn.grid(row=3, column=0, columnspan=2)
    # 载入数据集
    reDraw.rawDat = np.mat(regTrees.loadDataSet('sine.txt'))
    # 构造测试集
    reDraw.testDat = np.arange(np.min(reDraw.rawDat[:, 0]), np.max(reDraw.rawDat[:, 0]), 0.01)
    # 使用默认参数绘制
    reDraw(1.0, 10)
    root.mainloop()