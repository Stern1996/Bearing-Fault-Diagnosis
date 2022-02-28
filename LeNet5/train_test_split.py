#该程序用于按照label.txt随机划分一定比例训练集与测试集，并将划分好的数据分别存入对应文件夹下，此处选取20000张训练集，4000张图片作为测试集
import os
import random
import numpy as np

'''
f=open("label.txt")
all_lines=f.readlines()
test=random.sample(all_lines,4000)
all_lines=set(all_lines)
test=set(test)
train=all_lines-test
with open('ead_test.txt','a') as test_txt:
    for test_name in test:
        test_txt.write(test_name)
with open('ead_train.txt','a') as train_txt:
    for train_name in train:
        train_txt.write(train_name)  

'''

Images_path = "C:\\Users\\Administrator\\PycharmProjects\DA\\images\\"
train_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\train_dataset\\"  #you have to firstly create the directory
test_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\test_dataset\\"
with open("ead_train.txt","r") as f:
    trainfile_list = f.readlines()
    for name in trainfile_list:
        i = np.load((Images_path+name[0:-3]))
        with open(train_path+name[0:-3],'wb') as f:
            np.save(f,i)
with open("ead_test.txt","r") as f:
    trainfile_list = f.readlines()
    for name in trainfile_list:
        i = np.load((Images_path + name[0:-3]))
        with open(test_path+name[0:-3], 'wb') as f:
            np.save(f, i)


