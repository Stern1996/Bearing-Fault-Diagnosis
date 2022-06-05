import numpy as np
import scipy.io as scio
import random
import os

#该模块用来对CWRU原始数据进行采样并分类存储为npy文件，对于SVM分类，可调节单个样本所含数据点数，默认2048，每个mat文件抽取50个样本
#随后将所有采样样本数据整合成数据集，保存为npy文件，数据集shape为样本数x2048,标签格式样本数x类型标号
#请预先将所有CWRU要使用的训练集mat文件全放入dataset文件夹中!!!!
#This programm is used to translate the datafile from .mat to .npy and write the corresponding label for each file.
#The fault type of datafile(.mat) ist related to the filenames.


#The number of each type: no--0, ir--1, ba--2, or--3
sample_path = "./sampling"
all_data = []
all_labels = []
#Filename parameters
start_file_num = 96
end_file_num = 243
#All datafiles should be stored in this path before running this program.
path = "./dataset/"
#parameters
sample_len = int(input("Please set the length of each sample(number of points):"))
num_samples = int(input("Please set the number of samples for each mat-file:"))

#processing each datafile and generate the npy file, output[num_samples*(samplingrate*sampling_time)]
def samples_get(filename,sample_len,num_samples):
    # readin data
    data = scio.loadmat("./dataset/"+filename)
    num = filename.strip(".mat")
    if int(num) < 100:
        index = "X" + "0" + num + "_DE_time"
    else:
        index = "X" + num + "_DE_time"
    data = data[index]
    data = data.flatten()

    # set number for each fault type,0-NO, 1-IR, 2-BA, 3-OR
    if int(num) <= 100:
        typ = 0
    elif 108 < int(num) < 113 or 173 < int(num) < 178 or 212 < int(num) < 218:
        typ = 1
    elif 121 < int(num) < 126 or 188 < int(num) < 193 or 225 < int(num) < 230:
        typ = 2
    elif 134 < int(num) < 139 or 200 < int(num) < 205 or 237 < int(num) < 242:
        typ = 3

    # get the data
    samples = []
    for i in range(num_samples):
        start = random.randint(0, (data.shape[0] - int(sample_len) - 1))
        segment = data[start:(start + int(sample_len))]
        samples.append(segment)
    res = {filename: {typ: samples}}
    np.save("./sampling/"+num + "_" + str(typ) + ".npy", res)

#174.mat is an exception!!!!!
def exc_get(filename,sample_len,num_samples):
    # readin data
    data = scio.loadmat("./dataset/"+filename)
    num = filename.strip(".mat")
    index = "X" + "173" + "_DE_time"
    data = data[index]
    data = data.flatten()

    # set number for each fault type,0-NO, 1-IR, 2-BA, 3-OR
    if int(num) <= 100:
        typ = 0
    elif 108 < int(num) < 113 or 173 < int(num) < 178 or 212 < int(num) < 218:
        typ = 1
    elif 121 < int(num) < 126 or 188 < int(num) < 193 or 225 < int(num) < 230:
        typ = 2
    elif 134 < int(num) < 139 or 200 < int(num) < 205 or 237 < int(num) < 242:
        typ = 3

    # get the data
    samples = []
    for i in range(num_samples):
        start = random.randint(0, (data.shape[0] - int(sample_len) - 1))
        segment = data[start:(start + int(sample_len))]
        samples.append(segment)
    res = {filename: {typ: samples}}
    np.save("./sampling/"+num + "_" + str(typ) + ".npy", res)

def samples_concat():
    # save the data in different types
    files = os.listdir(sample_path)
    for file in files:
        data = np.load(sample_path + "/" + file, allow_pickle=True)
        data = data.item()
        mat_name = list(data.keys())[0]
        typ = list(data[mat_name].keys())[0]
        label = [typ] * len(data[mat_name][typ])
        data = data[mat_name][typ]

        global all_data
        global all_labels
        all_data.append(data)
        all_labels.append(label)

    all_data = np.asarray(all_data)
    all_labels = np.asarray(all_labels)

    # 降维
    all_data = all_data.reshape((-1, all_data.shape[2]))
    all_labels = all_labels.reshape((-1, 1))

    # the datasets shape: [采样文件数x每个文件样本数x2048]，标签[采样文件数x每个文件样本数x1]
    np.save("data/database.npy", all_data)
    np.save("data/labels.npy", all_labels)

if __name__ == "__main__":
    names = list(range(start_file_num,end_file_num))
    for name in names:
        if os.path.exists(path+str(name)+".mat") and name != 174:
            samples_get(str(name)+".mat",sample_len,num_samples)
        if os.path.exists(path+str(name)+".mat") and name == 174:
            exc_get('174.mat',sample_len,num_samples)
    #create the dataset and labelset
    samples_concat()