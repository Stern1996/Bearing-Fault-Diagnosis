#该程序用于将mat数据文件读入并通过STFT方法提取转换为时频图像并作灰度处理后保存
import numpy
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os
from scipy.signal import stft
import add_gaussnoise

no_datapath = "E:\\CWRU数据集\\STFT\\NO_noise5"
ba_datapath = "E:\\CWRU数据集\\STFT\\BA_noise5"
ir_datapath = "E:\\CWRU数据集\\STFT\\IR_noise5"
or_datapath = "E:\\CWRU数据集\\STFT\\OR_noise5"

def DataAcquisition(DirPath):
  files = os.listdir(DirPath)
  #提取所有该目录数据并整合
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
  #将获取的一维数据点分割抽取STFT图像
  flag_list = random.sample(list(range(0,(len(D_data)-1-96*96))),2819)   #randomly selected 2819 image samples from the data,totally 2819
  image_list = [] # create a list to store the sampling images
  image = numpy.zeros((96,96))
  for i in range(2819):
    segment = D_data[flag_list[i]:(flag_list[i]+9216)]
    f, t, Zxx = stft(segment, window='hamming', nperseg=190) #stft转换
    Zxx = np.abs(Zxx)
    Zxx = Zxx.flatten()
    for x in range(96):
      for y in range(96):
        image[x,y] = round((Zxx[x*96+y]-Zxx.min())/(Zxx.max()-Zxx.min())*255) #translate the data to the number in 0-255 to get gray-image
    image_list.append(image)
  return image_list

#建立对应轴承故障文件夹并存储相应的图片
if not os.path.exists("or_noise5"):
    os.makedirs("or_noise5")
or_images = DataAcquisition(or_datapath)
for w in list(range(2819)):
    n = 0
    im = Image.fromarray(or_images[w])
    n += 1
    im = im.convert("L")
    im.save(("./or_noise5"+"/"+("or"+"_"+str(w)+".png")))