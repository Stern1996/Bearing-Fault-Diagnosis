#SVM的模型框架结构
#coding: utf-8
from keras import regularizers

import tfrecord_generate
import numpy as np
from keras.layers import *
from keras.models import *
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
from model_profiler import model_profiler
import tensorflow as tf
from keras_flops import get_flops



train_segments, train_labels,validation_segments, validation_labels = tfrecord_generate.get_tfrecord()
#一维数据点升维为二维数据，所有点取相同X值，以Y值作为振动信号
train_segments = train_segments.flatten().reshape(20000*2048,1)
#考虑振动信号方向仅值作为特征识别，故此处取绝对值
train_segments = np.absolute(train_segments)
#升维
train_segments = np.concatenate((np.zeros((train_segments.shape[0],1)),train_segments),axis=1)
#标签从one-hot还原
train_labels = train_labels.repeat([2048],axis=0)
train_labels = np.argmax(train_labels,axis=1)

validation_segments = validation_segments.flatten().reshape(800*2048,1)
validation_segments = np.absolute(validation_segments)
validation_segments = np.concatenate((np.zeros((validation_segments.shape[0],1)),validation_segments),axis=1)
validation_labels = validation_labels.repeat([2048],axis=0)
validation_labels = np.argmax(validation_labels,axis=1)

#模型搭建
model = Sequential()

model.add(Dense(4, kernel_regularizer=regularizers.l2(0.01)))
model.add(Activation('linear'))
model.compile(loss='squared_hinge',
              optimizer='adadelta',
              metrics=['accuracy'])
#模型训练
history = model.fit(train_segments,train_labels,epochs=3,batch_size=4096,validation_split=0.2,shuffle=True)
model.evaluate(validation_segments,validation_labels)

'''
#训练过程可视化
fig = plt.figure()#新建一张图
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('1DCNN'+"20000_noise10"+'acc.png')  #视情况修改
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('1DCNN'+"20000_noise10"+'loss.png')  #视情况修改
'''
