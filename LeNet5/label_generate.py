#该程序用于根据2400张图片数据生成相对应的label.txt并给每个数据标上标签，如“014o_80 1”表示为第二类故障的图片
#该程序用于根据stft后灰度图片数据生成相对应的label.txt并给每个数据标上标签，分为10类
import os
import numpy as np
dir_list = ["NO","BA7","BA14","BA21","IR7","IR14","IR21","OR7","OR14","OR21"]
filepath = "C:\\Users\\Administrator\\PycharmProjects\\DA\\"  #the directory path of datasets
filenames = []
for i in dir_list:
    filename = os.listdir(filepath+i)
    filenames.append(filename)
filenames = [item for l in filenames for item in l]

#根据文件名称建立对应的标签文件
with open("label.txt","w") as f:
    for name in filenames:
        if name[0:3] == "no_":
            f.writelines(name+" "+"0"+"\n")
        elif name[0:3] == "ba7":
            f.writelines(name + " " + "1" + "\n")
        elif name[0:3] == "ba1":
            f.writelines(name + " " + "2" + "\n")
        elif name[0:3] == "ba2":
            f.writelines(name + " " + "3" + "\n")
        elif name[0:3] == "ir7":
            f.writelines(name + " " + "4" + "\n")
        elif name[0:3] == "ir1":
            f.writelines(name + " " + "5" + "\n")
        elif name[0:3] == "ir2":
            f.writelines(name + " " + "6" + "\n")
        elif name[0:3] == "or7":
            f.writelines(name + " " + "7" + "\n")
        elif name[0:3] == "or1":
            f.writelines(name + " " + "8" + "\n")
        elif name[0:3] == "or2":
            f.writelines(name + " " + "9" + "\n")