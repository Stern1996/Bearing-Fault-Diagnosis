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
from get_dataset import read_datasets


datasets, labels = read_datasets()
#数据集划分
X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2,random_state=0)
#数据预处理
X_train = np.real(X_train)
y_train = y_train.flatten()
X_test = np.real(X_test)
y_test = y_test.flatten()

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