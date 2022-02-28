# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 01:54:50 2022

@author: hailo
"""

import h5py
import tensorflow as tf
import numpy as np
from PIL import Image
import os

segments_train_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\train_dataset\\"
label_train_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\ead_train.txt"
h5_train = './data/segment_train.h5'
segments_test_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\test_dataset\\"
label_test_path = "C:\\Users\\Administrator\\PycharmProjects\\DA\\ead_test.txt"
h5_test = './data/segment_test.h5'
data_path = "./data"


def read_h5(filename):
    dataset = h5py.File(filename, "r")
    images = np.array(dataset["images"][:],dtype = "float64") # your train set features
    images = images.reshape(images.shape[0],images.shape[1],images.shape[2],1)
    labels = np.array(dataset["labels"][:],dtype = "int64") # your train set features
    labels_one_hot = np.zeros((labels.size, labels.max()+1))
    labels_one_hot[np.arange(labels.size),labels] = 1
    return images, labels_one_hot

def get_dataset():  #批量化读取
    train_images, train_labels = read_h5(h5_train)
    validation_images, validation_labels = read_h5(h5_test)
    return train_images, train_labels, validation_images, validation_labels


def generate_h5():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully!")
    else:
        print("directory already exists!")
    write_h5(h5_train, segments_train_path, label_train_path)
    write_h5(h5_test, segments_test_path, label_test_path)

def write_h5(h5Name, image_path, label_path):
    hf = h5py.File(h5Name, "w")
    f = open(label_path,'r')
    contents = f.readlines()
    f.close()
    img_data = np.zeros((len(contents), 64,64))
    label_data = np.zeros(len(contents))
    for i in range(len(contents)):
        value = contents[i].split()
        img_path = image_path + value[0]
        segment = np.load(img_path)
        label = int(value[1])
        img_data[i] = np.squeeze(segment)
        label_data[i] =  label   
        print("the number of picture:", i)
    
    hf.create_dataset('images', data=img_data)
    hf.create_dataset('labels', data=label_data)
    hf.close()
    print("write h5 successfully!")

def main():
    generate_h5()

if __name__ == '__main__':
    main()