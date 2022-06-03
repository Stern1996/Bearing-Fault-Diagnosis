#SVM的模型框架结构
#coding: utf-8
from keras import regularizers

import numpy as np
from keras.layers import *
from keras.models import *
from sklearn import svm
import joblib
from sklearn.multiclass import OneVsOneClassifier
from sklearn.model_selection import train_test_split

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
    acc = clt.score(X_test,y_test)
    print("The accuracy of model is:",acc)

#注意，输入用于模型预测的数据结构为ndarray(样本数x2048),2048表示每个样本数据点数
def model_predict(testdata):
    model = joblib.load("svm_4.pkl")
    res = model.predict(testdata)
    return res

if __name__ == "__main__":
    model_train()
    #res = model_predict(testdata)