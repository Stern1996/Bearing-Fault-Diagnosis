#该程序用于按照label.txt随机划分一定比例训练集与测试集(7:3)，并将划分好的数据分别存入对应文件夹下,需要提前手动将四个文件夹的图片数据存入一个文件夹中
import os
import random
from PIL import Image
def label_split():  #对比所有文件从文件名上进行划分
    f=open("label.txt")
    all_lines=f.readlines()
    test=random.sample(all_lines,int(len(all_lines)*0.3))
    all_lines=set(all_lines)
    test=set(test)
    train=all_lines-test
    with open('ead_test.txt','a') as test_txt:
        for test_name in test:
            test_txt.write(test_name)
    with open('ead_train.txt','a') as train_txt:
        for train_name in train:
            train_txt.write(train_name)

def image_split(): #对图片数据进行划分并存储
    Images_path = "C:\\Users\\Administrator\\PycharmProjects\\TFCNN\\images\\"
    train_path = "C:\\Users\\Administrator\\PycharmProjects\\TFCNN\\train_dataset\\"
    test_path = "C:\\Users\\Administrator\\PycharmProjects\\TFCNN\\test_dataset\\"
    with open("ead_train.txt","r") as f:
        trainfile_list = f.readlines()
        for name in trainfile_list:
            i = Image.open((Images_path+name[0:-3]))
            i.save((train_path+name[0:-3]))
    with open("ead_test.txt","r") as f:
        trainfile_list = f.readlines()
        for name in trainfile_list:
            i = Image.open((Images_path+name[0:-3]))
            i.save((test_path+name[0:-3]))

if __name__ == '__main__':
    label_split()
    image_split()