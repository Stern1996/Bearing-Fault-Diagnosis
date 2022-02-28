#该程序用于将包含振动信号的mat文件数据集提取,1000个时间窗，四类故障，每个时间窗包含2048个数据点，保存.npy数据和对应labels
import numpy
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os

no_datapath = "E:\\CWRU数据集\\STFT\\NO_noise"
ba_datapath = "E:\\CWRU数据集\\STFT\\BA_noise"
ir_datapath = "E:\\CWRU数据集\\STFT\\IR_noise"
or_datapath = "E:\\CWRU数据集\\STFT\\OR_noise"


#讲一维数据分段隔出并存储至ndarray中[samples,lenth of segment,1]
def DataAcquisition(DirPath,label): #给定类别标号，0代表第一类，1代表第二类，以此类推
  files = os.listdir(DirPath)
  # 提取所有该目录数据并整合
  all_data = []
  for file in files:
    data = scio.loadmat(DirPath + "\\" + file)
    num = file.strip(".mat")
    if int(num) < 100:
      index = "X" + "0" + num + "_DE_time"
    else:
      index = "X" + num + "_DE_time"
    data = data[index]
    data = np.asarray(data)
    data = data.flatten()
    data = list(data)
    all_data.append(data)
  D_data = []
  for i in all_data:
    D_data += i
  #将添加噪音后的一维数据点分割抽取
  flag_list = random.sample(list(range(0,(len(D_data)-1-2048))),1000)   #randomly selected 5200 samples from the data,include training5000 and validation200
  segment_list = []
  for i in range(1000):
    segment = D_data[flag_list[i]:(flag_list[i]+2048)]
    segment_list.append(segment)
  labels = np.asarray([label]*(1000*2048))
  segment_list = np.asarray(segment_list).flatten()
  #构建时间坐标，最后给SVM训练数据应为2维坐标
  t = np.linspace(1,1000*2048,1000*2048)
  segment_list = np.vstack((t,segment_list)).T
  return segment_list, labels

def read_datasets():
  dataset_no = np.load('./NO/no.npy')
  labels_no = np.load('./NO/no_labels.npy').reshape((2048*1000,1))
  dataset_ba = np.load('./BA/ba.npy')
  labels_ba = np.load('./BA/ba_labels.npy').reshape((2048*1000,1))
  dataset_ir = np.load('./IR/ir.npy')
  labels_ir = np.load('./IR/ir_labels.npy').reshape((2048*1000,1))
  dataset_or = np.load('./OR/or.npy')
  labels_or = np.load('./OR/or_labels.npy').reshape((2048*1000,1))
  train_datasets = np.vstack((dataset_no,dataset_ba,dataset_ir,dataset_or))
  labels = np.vstack((labels_no,labels_ba,labels_ir,labels_or))
  return train_datasets, labels

if not os.path.exists("OR"):
    os.makedirs("OR")
segments, labels = DataAcquisition(or_datapath,3)
np.save("./OR/or.npy",segments)
np.save("./OR/or_labels.npy",labels)
