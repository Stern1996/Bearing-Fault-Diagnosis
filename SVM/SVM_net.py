#SVM的模型框架结构
#coding: utf-8
#SVM model
#coding: utf-8
import os
import pickle

from keras import regularizers

import numpy as np
from keras.layers import *
from keras.models import *
from sklearn import svm
import joblib
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

#only use CPU to run the program
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

def model_train():
    #每个样本为（2048，）
    datasets, labels = np.load('./data/database.npy'), np.load('./data/labels.npy')
    #数据集划分
    X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2,random_state=0)
    y_train, y_test = y_train.flatten(), y_test.flatten()

    #模型搭建与训练
    model = OneVsOneClassifier(svm.SVC(C=1.5,kernel='rbf',))
    print("[INFO] Successfully initialize a new model !")
    print("[INFO] Training the model…… ")
    clt = model.fit(X_train,y_train)
    print("[INFO] Model training completed !")
    # 保存训练好的模型，下次使用时直接加载就可以了
    joblib.dump(clt,"svm_4.pkl")
    print("[INFO] Model has been saved !")

    #输出训练模型的精度
    start_time = time.time()
    acc = clt.score(X_test,y_test)
    end_time = time.time()
    print("The predict time:",(end_time-start_time))
    print("The accuracy of model is:",acc)

#注意，输入用于模型预测的数据结构为ndarray(样本数x2048),2048表示每个样本数据点数
def model_predict(testdata):
    model = joblib.load("svm_4.pkl")
    datasets, labels = np.load('./data/MFPT_database.npy'), np.load('./data/MFPT_labels.npy')
    # dataset split
    X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.5, random_state=0)
    y_train, y_test = y_train.flatten(), y_test.flatten()
    clt = model.fit(X_train,y_train)
    start_time = time.time()
    res = clt.predict(X_test)
    end_time = time.time()
    print("Predict time is:",(end_time-start_time))
    cm = confusion_matrix(y_test, res)
    return cm

#绘制SVM模型预测结果的混淆矩阵（按需使用）
def plot_confusion_matrix(cm, savename, title='Confusion Matrix'):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=15, va='center', ha='center')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title,fontdict={'family' : 'Times New Roman', 'size'   : 20})
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90,fontproperties = 'Times New Roman', size = 16)
    plt.yticks(xlocations, classes,fontproperties = 'Times New Roman', size = 16)
    plt.ylabel('Actual label',fontdict={'family':'Times New Roman', 'size':18})
    plt.xlabel('Predict label',fontdict={'family':'Times New Roman', 'size':18})

    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)

    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()


if __name__ == "__main__":
    model_train()

    #根据预测数据集的类别设置
    classes = ['0', '1', '2', '3']

    #如需使用训练好的模型预测并绘制混淆矩阵，使用如下代码
    # cm = model_predict()
    # plot_confusion_matrix(cm, 'Lenet_confusion_matrix.png', title='confusion matrix')