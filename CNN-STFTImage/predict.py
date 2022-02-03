import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as plt
import random
from scipy.ndimage import zoom


#随机从验证集中选取100个样本作为测试，评估模型的精度,一共测试10轮
f=open("C:\\Users\\Administrator\\PycharmProjects\\TFCNN\\ead_test.txt")
all_lines=f.readlines()
new_model = tf.keras.models.load_model('stft.h5')
testdata_path = "C:\\Users\\Administrator\\PycharmProjects\\TFCNN\\test_dataset\\"
for x in range(10):
    grads = float()
    test = random.sample(all_lines, 100)
    for test_name in test:
        img = Image.open(testdata_path+test_name[0:-3])
        img = np.asarray(img)
        img = img.reshape(1, 96, 96, 1)
        #img = zoom(img,[1,32/96,32/96,1],order=1)
        img = img / 255
        predict = np.argmax(new_model.predict(img))
        if int(predict) == int(test_name[-2]):
            grads += 1
    test_acc = "{0} epoch: The validation accuracy is：{1}%".format(x+1,grads)
    with open("validation_acc.txt","a") as f:
        f.writelines(test_acc+"\n")