#该程序用于根据stft后灰度图片数据生成相对应的label.txt并给每个数据标上标签，仅分为N，BF,IR,OR四类
import os
import numpy as np
dir_list = ["no_noise5","ba_noise5","ir_noise5","or_noise5"]
filepath = "C:\\Users\\Administrator\\PycharmProjects\\TFCNN\\"  #the directory path of datasets
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