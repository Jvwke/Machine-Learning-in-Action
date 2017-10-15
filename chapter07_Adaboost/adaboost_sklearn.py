import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from chapter07_Adaboost import adaColic


# see: http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
if __name__ == '__main__':
    Xtrain, ytrain = adaColic.loadDataSet('horseColicTraining2.txt')
    Xtest, ytest = adaColic.loadDataSet('horseColicTest2.txt')
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm='SAMME', n_estimators=10)
    clf.fit(Xtrain, ytrain)
    predictions = clf.predict(Xtrain)
    errArr = np.mat(np.ones((len(Xtrain), 1)))
    print('training set error rate: %.3f%%' % (float(errArr[predictions != ytrain].sum()) / len(Xtrain) * 100.0))
    predictions = clf.predict(Xtest)
    errArr = np.mat(np.ones((len(Xtest), 1)))
    print('test set error rate: %.3f%%' % (float(errArr[predictions != ytest].sum()) / len(Xtest) * 100.0))
