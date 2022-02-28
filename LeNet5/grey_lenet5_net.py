#lenet5的模型框架结构
#coding: utf-8
import greyImage_generate
from keras.layers import *
from keras.models import *
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.optimizers import Adam
from model_profiler import model_profiler
from keras_flops import get_flops
import h5_generate


train_images, train_labels,validation_images, validation_labels =h5_generate.get_dataset()


#搭建模型
model = Sequential()
#Layer 1
#Conv Layer 1
model.add(Conv2D(filters=32,kernel_size=5,strides=1,activation='relu',padding='same',input_shape=(64,64,1)))
#Pooling layer 1
model.add(MaxPooling2D(pool_size=2,strides=2))
#Layer 2
#Conv Layer 2
model.add(Conv2D(filters=64,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(32,32,32)))
#Pooling layer 2
model.add(MaxPooling2D(pool_size=2,strides=2))
#Layer 3
#Conv Layer 3
model.add(Conv2D(filters=128,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(16,16,64)))
#Pooling layer 3
model.add(MaxPooling2D(pool_size=2,strides=2))
#Layer 4
#Conv Layer 4
model.add(Conv2D(filters=256,kernel_size=3,strides=1,activation='relu',padding='same',input_shape=(8,8,128)))
#Pooling layer 4
model.add(MaxPooling2D(pool_size=2,strides=2))
#Flatten
model.add(Flatten())
#Layer 5
#Fully connected layer 1
model.add(Dense(units=2560,activation='relu'))
#model.add(Dense(units=768,activation='relu'))
#Layer 6
#output Layer
model.add(Dense(units=10,activation='softmax'))
model.summary()
adam = Adam(learning_rate=0.0001,beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.0,amsgrad=False)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

flops = get_flops(model,batch_size=32)
flops = flops/2
print(f"FLOPS: {flops / 10 ** 9:.03} G")

history = model.fit(train_images,train_labels,epochs=10,batch_size=32,validation_split=0.2)
model.evaluate(validation_images,validation_labels)
model.save('grey.h5')

#训练过程可视化
fig = plt.figure()#新建一张图
plt.plot(history.history['accuracy'],label='training acc')
plt.plot(history.history['val_accuracy'],label='val acc')
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='lower right')
fig.savefig('LeNet5'+"24000"+'acc.png')
fig = plt.figure()
plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')
fig.savefig('LeNet5'+"24000"+'loss.png')
