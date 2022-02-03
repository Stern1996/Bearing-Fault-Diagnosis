from keras.models import load_model, Model
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from sklearn.manifold import TSNE

load_model = load_model('grey.h5')
model = Model(inputs=load_model.input, outputs=load_model.get_layer('dense_1').output)
f=open("C:\\Users\\Administrator\\PycharmProjects\\DA\\ead_test.txt")
all_lines=f.readlines()
testdata_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\test_dataset\\"

img_list = []
for x in range(10):
    grads = float()
    test = random.sample(all_lines, 100)
    for test_name in test:
        img = Image.open(testdata_path+test_name[0:-3])
        img = np.asarray(img)
        img = img.reshape(64,64,1)
        img = img / 255
        img_list.append(img)
img_list = np.asarray(img_list)
print(img_list.shape)

pres = np.zeros(shape=(1000,1,1,4))
label = np.zeros(shape=(1000,))
for i in range(1000):
    predict = model.predict(img_list[i].reshape(1,64,64,1))
    cl = np.argmax(predict)
    label[i] = int(cl)
    pres[i] = predict

X= pres.reshape(1000,4)
tsne = TSNE(n_components=2,learning_rate=1000,init='pca',random_state=0)
X_tsne = tsne.fit_transform(X)

x_max = int(max(X_tsne[ : , :1]))
x_min = int(min(X_tsne[ : , :1]))
y_max = int(max(X_tsne[ : ,1:2]))
y_min = int(min(X_tsne[ : ,1:2]))

print('x_max:',x_max, 'x_min:',x_min,'y_max:',y_max,'y_min:',y_min)

#画出分类散点图
type1_x = []
type1_y = []
type2_x = []
type2_y = []
type3_x = []
type3_y = []
type4_x = []
type4_y = []

for j in range(1000):
    if label[j] == 0:
        type1_x.append(X_tsne[j,0])
        type1_y.append(X_tsne[j,1])
    elif label[j] == 1:
        type2_x.append(X_tsne[j, 0])
        type2_y.append(X_tsne[j, 1])
    elif label[j] == 2:
        type3_x.append(X_tsne[j, 0])
        type3_y.append(X_tsne[j, 1])
    else:
        type4_x.append(X_tsne[j, 0])
        type4_y.append(X_tsne[j, 1])

plt.scatter(type1_x, type1_y, c='r',label='class-1')
plt.scatter(type2_x, type2_y, c='g',label='class-2')
plt.scatter(type3_x, type3_y, c='k',label='class-3')
plt.scatter(type4_x, type4_y, c='b',label='class-4')

plt.xticks(np.linspace(x_min,x_max,10))
plt.yticks(np.linspace(y_min,y_max,10))
plt.legend()
plt.savefig('lenet5_features.png')
plt.show()