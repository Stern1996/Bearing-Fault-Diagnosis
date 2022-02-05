#该程序用于根据20800个数据段生成相对应的label.txt并给每个数据标上标签，如“ba_3393.npy 1”表示为第二类故障的图片
#仅分为N，BF,IR,OR四类
import os
import numpy as np
dir_list = ["no_noise10","ba_noise10","ir_noise10","or_noise10"]
filepath = "C:\\Users\\Administrator\\PycharmProjects\\DA_testmodel\\SVM\\"  #the directory path of datasets
filenames = []
for i in dir_list:
    filename = os.listdir(filepath+i)
    filenames.append(filename)
filenames = [item for l in filenames for item in l]

#根据文件名称建立对应的标签文件
with open("label.txt","w") as f:
    for name in filenames:
        if name[0:2] == "no":
            f.writelines(name+" "+"0"+"\n")
        elif name[0:2] == "ba":
            f.writelines(name + " " + "1" + "\n")
        elif name[0:2] == "ir":
            f.writelines(name + " " + "2" + "\n")
        elif name[0:2] == "or":
            f.writelines(name + " " + "3" + "\n")
