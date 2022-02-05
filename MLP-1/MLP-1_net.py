#MLP的模型框架结构(1层隐藏层)
#coding: utf-8

import greyImage_generate
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
from model_profiler import model_profiler
import tensorflow as tf
from keras_flops import get_flops



train_segments, train_labels,validation_segments, validation_labels = greyImage_generate.get_tfrecord()
#由于数据集中各个数据值的差别不大，故不需要进行数据预处理

#搭建模型
model = Sequential()

model.add(Dense(units=512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=4,activation='softmax'))
model.summary()
adam = Adam(learning_rate=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

flops = get_flops(model,batch_size=32)
flops = flops/2
print(f"FLOPS: {flops / 10 ** 9:.03} G")

history = model.fit(train_segments,train_labels,epochs=10,batch_size=32,validation_split=0.2,shuffle=True)
model.evaluate(validation_segments,validation_labels)
model.save('grey.h5')


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

