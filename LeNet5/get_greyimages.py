#该程序用于将包含振动信号的mat文件数据集转化为二维图片并分类存储
import numpy
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os

no_datapath = "E:\\CWRU数据集\\STFT\\type10\\NO"
ba7_datapath = "E:\\CWRU数据集\\STFT\\type10\\BA7"
ba14_datapath = "E:\\CWRU数据集\\STFT\\type10\\BA14"
ba21_datapath = "E:\\CWRU数据集\\STFT\\type10\\BA21"
ir7_datapath = "E:\\CWRU数据集\\STFT\\type10\\IR7"
ir14_datapath = "E:\\CWRU数据集\\STFT\\type10\\IR14"
ir21_datapath = "E:\\CWRU数据集\\STFT\\type10\\IR21"
or7_datapath = "E:\\CWRU数据集\\STFT\\type10\\OR7"
or14_datapath = "E:\\CWRU数据集\\STFT\\type10\\OR14"
or21_datapath = "E:\\CWRU数据集\\STFT\\type10\\OR21"

def DataAcquisition(DirPath):
  files = os.listdir(DirPath)
  # 提取所有该目录数据并整合
  all_data = []
  D_data = []
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
    #data = add_gaussnoise.add_noise(data)
    data = list(data)
    all_data.append(data)
  for i in all_data:
    D_data += i

  #将添加噪音后的一维数据点分割抽取转化为灰度图
  flag_list = random.sample(list(range(0,(len(D_data)-1-64*64))),2400)   #randomly selected 2400 image samples from the data
  image_list = [] # create a list to store the sampling images
  for i in range(2400):
    segment = D_data[flag_list[i]:(flag_list[i]+64*64)]
    segment = np.asarray(segment)
    segment = (segment-min(segment))/(max(segment)-min(segment))*255
    image = segment.reshape((64,64))
    image_list.append(image)
  return image_list



if not os.path.exists("OR21"):
    os.makedirs("OR21")
or21_images = DataAcquisition(or21_datapath)
or21_images = numpy.asarray(or21_images)
for i in range(2400):
  name = "./OR21"+"/"+("or21"+"_"+str(i))+".npy"
  with open(name,'wb') as f:
    np.save(f,or21_images[i])
'''
for w in list(range(2400)):
    n = 0
    im = Image.fromarray(or21_images[n])
    n += 1
    im = im.convert("L")
    im.save(("./OR21"+"/"+("or21"+"_"+str(w)+".png")))
'''