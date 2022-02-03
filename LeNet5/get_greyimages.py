#该程序用于将包含振动信号的mat文件数据集转化为二维图片并分类存储
import numpy
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os
import add_gaussnoise

no_datapath = "E:\\CWRU数据集\\STFT\\NO_noise5"
ba_datapath = "E:\\CWRU数据集\\STFT\\BA_noise5"
ir_datapath = "E:\\CWRU数据集\\STFT\\IR_noise5"
or_datapath = "E:\\CWRU数据集\\STFT\\OR_noise5"

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
    #data = add_gaussnoise.add_noise(data)
    data = list(data)
    all_data.append(data)
  D_data = []
  for i in all_data:
    D_data += i
  #将添加噪音后的一维数据点分割抽取转化为灰度图
  flag_list = random.sample(list(range(0,(len(D_data)-1-64*64))),2400)   #randomly selected 2400 image samples from the data
  image_list = [] # create a list to store the sampling images
  image = numpy.zeros((64,64))
  for i in range(2400):
    segment = D_data[flag_list[i]:(flag_list[i]+64*64)]
    for x in range(64):
      for y in range(64):
        image[x,y] = round((segment[x*64+y]-min(segment))/(max(segment)-min(segment))*255) #translate the data to the number in 0-255 to get gray-image
    image_list.append(image)
  return image_list

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


if not os.path.exists("or_noise5"):
    os.makedirs("or_noise5")
or_images = DataAcquisition(or_datapath)
for w in list(range(2400)):
    n = 0
    im = Image.fromarray(or_images[n])
    n += 1
    im = im.convert("L")
    im.save(("./or_noise5"+"/"+("or"+"_"+str(w)+".png")))