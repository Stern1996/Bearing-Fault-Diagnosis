#论文Deep Learning Enabled Fault Diagnosis Using Time-Frequency Image Analysis of Rolling Element Bearings的模型框架结构
#coding: utf-8
import numpy as np

import tfrecord_generate
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
from scipy.ndimage import zoom
from model_profiler import model_profiler
from keras_flops import get_flops

train_images, train_labels,validation_images, validation_labels = tfrecord_generate.get_tfrecord()
#注意：在从tfrecord读取过程中，已完成了对图片的预处理工作，所以这里不用再进行额外处理
train_images = train_images.reshape((8228,96,96,1))   #可从划分的图片文件夹中得知图片的数量
validation_images = validation_images.reshape((3525,96,96,1))

'''
resize_width = 96
resize_height = 96
new_train_images = np.zeros((8228,resize_width,resize_height,1))
new_validation_images = np.zeros((3525,resize_width,resize_height,1))

#依据论文所选插值方法对图片进行缩放以适应对应的CNN网络输入，文中选择bilinear interpolation，故此order=1
for i in range(8228):
    new_train_images[i] = zoom(train_images[i],[resize_width/96,resize_height/96,1],order=1)
for m in range(3525):
    new_validation_images[m] = zoom(validation_images[m],[resize_width/96,resize_height/96,1],order=1)
train_images = new_train_images
validation_images = new_validation_images

'''

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


history = model.fit(train_images,train_labels,epochs=10,batch_size=32,validation_split=0.2)
model.evaluate(validation_images,validation_labels)
model.save('stft.h5')

#训练过程可视化
fig = plt.figure()#新建一张图
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('stft'+"8228"+'acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('stft'+"8228"+'loss.png')
