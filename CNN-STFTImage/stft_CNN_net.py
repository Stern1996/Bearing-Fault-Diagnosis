#论文Deep Learning Enabled Fault Diagnosis Using Time-Frequency Image Analysis of Rolling Element Bearings的模型框架结构
#coding: utf-8
import numpy as np
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.optimizers import Adam
from keras_flops import get_flops

# 每个样本为（64，64）
datasets, labels = np.load('./data/database.npy'), np.load('./data/labels.npy')
#标签集one-hot编码,此时labels为（样本数x4）
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
datasets = datasets.reshape((datasets.shape[0],datasets.shape[1],datasets.shape[2],1))
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=0)

#搭建模型
model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters=32,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(96,96,1)))
#Conv Layer 2
model.add(Conv2D(filters=32,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(96,96,32)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size=2,strides=2))
#Layer 2
#Conv Layer 3
model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(48,48,32)))
#Conv Layer 4
model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(48,48,64)))
#Pooling layer 2
model.add(MaxPooling2D(pool_size=2,strides=2))
#Layer 3
#Conv Layer 5
model.add(Conv2D(filters=128,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(24,24,64)))
#Conv Layer 6
model.add(Conv2D(filters=128,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(24,24,128)))
#Pooling layer 3
model.add(MaxPooling2D(pool_size=2,strides=2))
#Flatten
model.add(Flatten())
#Layer 4
#Fully connected layer 1
model.add(Dense(units=100,activation='relu'))
#Fully connected layer 2
model.add(Dense(units=100,activation='relu'))
#Layer 5
#output Layer
model.add(Dense(units=4,activation='softmax'))
model.summary()
adam = Adam(learning_rate=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

flops = get_flops(model,batch_size=32)
flops = flops/2
print(f"FLOPS: {flops / 10 ** 9:.03} G")


history = model.fit(X_train,y_train,epochs=150,batch_size=16,validation_split=0.2)
model.evaluate(X_test,y_test)
model.save('stft_CNN.h5')

#训练过程可视化
fig = plt.figure()#新建一张图
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('stft_CNN_'+'acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('stft_CNN_'+'loss.png')
