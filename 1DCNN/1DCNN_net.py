#lenet5的模型框架结构
#coding: utf-8

import numpy as np
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from model_profiler import model_profiler
from sklearn.preprocessing import OneHotEncoder
from keras_flops import get_flops



#每个样本为（2048，）
datasets, labels = np.load('./data/database.npy'), np.load('./data/labels.npy')
#标签集one-hot编码,此时labels为（样本数x4）
enc = OneHotEncoder()
enc.fit(labels)
labels = enc.transform(labels).toarray()
#数据集划分
X_train, X_test, y_train, y_test = train_test_split(datasets, labels, test_size=0.2,random_state=0)

#搭建模型
model = Sequential()


#1
model.add(Conv1D(filters=16,kernel_size=128,strides=32,padding='same',activation='relu',input_shape=(2048,1)))
model.add(MaxPooling1D(pool_size=2))
#2
model.add(Conv1D(filters=16,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
#3
model.add(Conv1D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
#4
model.add(Conv1D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
#5
model.add(Conv1D(filters=32,kernel_size=3,strides=1,padding='same',activation='relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())
model.add(Dense(units=50,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4,activation='softmax'))
model.summary()
adam = Adam(learning_rate=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

flops = get_flops(model,batch_size=16)
flops = flops/2
print(f"FLOPS: {flops / 10 ** 9:.03} G")

'''
Batch_size = 32
profile = model_profiler(model, Batch_size,use_units=['GPU IDs', 'MFLOPs', 'MB', 'Million', 'MB'])

print(profile)

'''


history = model.fit(X_train,y_train,epochs=50,batch_size=16,validation_split=0.2,shuffle=True)
model.evaluate(X_test,y_test)
model.save('1DCNN.h5')


#训练过程可视化
fig = plt.figure()#新建一张图
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('1DCNN_'+'acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('1DCNN_'+'loss.png')

