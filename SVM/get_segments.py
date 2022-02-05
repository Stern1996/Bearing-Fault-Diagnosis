#该程序用于将包含振动信号的mat文件数据集提取,20800个样本，四类故障，每个样本包含2048个数据点
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
def DataAcquisition(DirPath):
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
  flag_list = random.sample(list(range(0,(len(D_data)-1-2048))),5200)   #randomly selected 5200 samples from the data,include training5000 and validation200
  segment_list = []
  for i in range(5200):
    segment = D_data[flag_list[i]:(flag_list[i]+2048)]
    segment_list.append(segment)
  return segment_list

# dia_list = ["7","14","21"]
# po_list = ["i","b","o"]
# load_list = ["0","1","2","3"]
# for x in load_list:
#   for y in dia_list:
#     for z in po_list:
#       dirs = x+"/"+y+"/"+z
#       if not os.path.exists(dirs):
#         os.makedirs(dirs)
#       images = DataAcquisition("./dataset/"+(x+y+z)+".mat")
#       for n in range(len(images)):
#         im = Image.fromarray(images[n])
#         im = im.convert("L")
#         im.save((dirs+"/"+(x+y+z)+"_"+str(n)+".png"))

# for m in load_list:
#   dirs = m
#   if not os.path.exists(dirs):
#     os.makedirs(dirs)
#   no_images = DataAcquisition("./dataset/"+m+"no.mat")
#   for w in range(len(no_images)):
#     im = Image.fromarray(no_images[w])
#     im = im.convert("L")
#     im.save((dirs+"/"+(m+"no"+"_"+str(w)+".png")))


if not os.path.exists("or_noise10"):
    os.makedirs("or_noise10")
segments = DataAcquisition(or_datapath)
segments = np.asarray(segments)
segments = np.expand_dims(segments,axis = 2)
for w in list(range(5200)):
    im = segments[w]
    np.save(("./or_noise10"+"/"+("or"+"_"+str(w)+".npy")),im)
