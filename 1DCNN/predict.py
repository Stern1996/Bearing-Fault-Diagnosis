import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib as plt
import random


#随机从验证集中选取100个样本作为测试，评估模型的精度
f=open("C:\\Users\\Administrator\\PycharmProjects\\DA_testmodel\\ead_test.txt")
all_lines=f.readlines()
new_model = tf.keras.models.load_model('grey.h5')
testdata_path = "C:\\Users\Administrator\\PycharmProjects\\DA_testmodel\\test_dataset\\"
for x in range(10):
    grads = float()
    test = random.sample(all_lines, 100)
    for test_name in test:
        segment = np.load(testdata_path+test_name[0:-3])
        segment = np.asarray(segment)
        segment = segment.reshape(1, 2048, 1)
        predict = np.argmax(new_model.predict(segment))
        if int(predict) == int(test_name[-2]):
            grads += 1
    test_acc = "{0} epoch: The validation accuracy is：{1}%".format(x+1,grads)
    with open("validation_half_noise_acc.txt", "a") as f:
        f.writelines(test_acc+"\n")

'''
img = Image.open("./221b_62.png")
img = np.asarray(img)
img = img.reshape(1,64,64,1)
img = img/255
new_model = tf.keras.models.load_model('grey.h5')
predict = new_model.predict(img)
print("correct class:7")
print("predict class:",str(np.argmax(predict)))
'''