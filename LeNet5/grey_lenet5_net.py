#lenet5的模型框架结构,keras搭建
#coding: utf-8
from sklearn.model_selection import train_test_split

import greyImage_generate
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.optimizers import Adam
from model_profiler import model_profiler
from keras_flops import get_flops
from sklearn.preprocessing import OneHotEncoder


# 每个样本为（64，64）
datasets, labels = np.load('./data/database.npy'), np.load('./data/labels.npy')
#标签集one-hot编码,此时labels为（样本数x4）
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2, random_state=0)


#搭建模型
model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters=32,kernel_size=5,strides=1,activation='relu',padding='same',input_shape=(64,64,1)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size=2,strides=2))
#Dropout1
model.add(Dropout(0.20))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(32,32,32)))
#Pooling layer 2
model.add(MaxPooling2D(pool_size=2,strides=2))
#Dropout2
model.add(Dropout(0.20))
#Layer 3
#Conv Layer 3
model.add(Conv2D(filters=128,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(16,16,64)))
#Dropout3
model.add(Dropout(0.20))
#Pooling layer 3
model.add(MaxPooling2D(pool_size=2,strides=2))
#Layer 4
#Conv Layer 4
model.add(Conv2D(filters=256,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(8,8,128)))
#Pooling layer 4
model.add(MaxPooling2D(pool_size=2,strides=2))
#Dropout4
model.add(Dropout(0.20))
#Flatten
model.add(Flatten())
#Layer 5
#Fully connected layer 1
model.add(Dense(units=84,activation='relu'))
#model.add(Dense(units=768,activation='relu'))
#Layer 6
#output Layer,标签为4类，所以units设定为4
model.add(Dense(units=4,activation='softmax'))
model.summary()
adam = Adam(learning_rate=0.00001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# flops = get_flops(model,batch_size=16)
# flops = flops/2
# print(f"FLOPS: {flops / 10 ** 9:.03} G")

history = model.fit(X_train,y_train,epochs=200,batch_size=32,validation_split=0.2)
model.evaluate(X_test,y_test)
model.save('LeNet5.h5')

#训练过程可视化
fig = plt.figure()
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy',fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel('accuracy',fontdict={'family':'Times New Roman', 'size':16})
plt.xlabel('epoch',fontdict={'family':'Times New Roman', 'size':16})
plt.legend(loc='lower right',prop={'family' : 'Times New Roman', 'size'   : 16})
fig.savefig('LeNet5_'+'acc.png')
fig = plt.figure()
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss',fontdict={'family' : 'Times New Roman', 'size'   : 16})
plt.ylabel('loss',fontdict={'family':'Times New Roman', 'size':16})
plt.xlabel('epoch',fontdict={'family':'Times New Roman', 'size':16})
plt.legend(loc='upper right',prop={'family' : 'Times New Roman', 'size'   : 16})
fig.savefig('LeNet5_'+'loss.png')
