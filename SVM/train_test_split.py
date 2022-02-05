#该程序用于按照label.txt随机划分一定比例训练集与测试集，并将划分好的数据分别存入对应文件夹下，此处选取20000个样本为训练集，800个为测试集
import os
import random
from PIL import Image
import numpy as np

#Firstly run this part

'''
f=open("label.txt")
all_lines=f.readlines()
test=random.sample(all_lines,800)
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

#then lock first part and run this part
Segments_path = "C:\\Users\\Administrator\\PycharmProjects\\DA_testmodel\\SVM\\segments\\"  #you should create and store all .npy-files in this directory manually
train_path = "C:\\Users\\Administrator\\PycharmProjects\\DA_testmodel\\SVM\\train_dataset\\"  #you have to firstly create the directory
test_path = "C:\\Users\\Administrator\\PycharmProjects\\DA_testmodel\\SVM\\test_dataset\\"
with open("ead_train.txt","r") as f:
    trainfile_list = f.readlines()
    for name in trainfile_list:
        i = np.load((Segments_path+name[0:-3]))
        np.save((train_path+name[0:-3]),i)
with open("ead_test.txt","r") as f:
    trainfile_list = f.readlines()
    for name in trainfile_list:
        i = np.load((Segments_path + name[0:-3]))
        np.save((test_path + name[0:-3]), i)

