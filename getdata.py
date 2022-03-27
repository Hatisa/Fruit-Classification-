import os
import torch.utils.data as data
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

IMAGE_SIZE=100
dataTransform=transforms.Compose([
    transforms.Resize(IMAGE_SIZE),                          # 将图像按比例缩放至合适尺寸
    transforms.CenterCrop((IMAGE_SIZE, IMAGE_SIZE)),        # 从图像中心裁剪合适大小的图像
    transforms.ToTensor()   # 转换成Tensor形式，并且数值归一化到[0.0, 1.0]，同时将H×W×C的数据转置成C×H×W，这一点很关键
])

class FRDataset(data.Dataset):
    def __init__(self,mode,dir):
        self.mode=mode
        self.list_img=[]
        self.list_label=[]
        self.data_size=0
        self.transform=dataTransform

        if self.mode =='train':
            dir = dir + 'train/'
            i = -1
            for file in os.listdir(dir):
                i = i+1
                name = file
                for img in os.listdir(dir + name):
                    self.list_img.append(dir + name + '/'+ img)
                    self.data_size += 1
                    self.list_label.append(i)
        elif self.mode=='test':
            dir=dir+'test/'
            for img in os.listdir(dir):
                self.data_size+=1
                self.list_label.append(33)  # 训练集
        else:
            print('Underfined Dataset!')

    def __getitem__(self, item):
        if self.mode=='train':
            img= Image.open(self.list_img[item])#打开图片
            label= self.list_label[item]#读取该图片label
            return self.transform(img),torch.LongTensor([label])
        elif self.mode=='test':
            img = Image.open(self.list_img[item])  # 打开图片
            return self.transform(img)
        else:
            print('None')

    def __len__(self):
        return self.data_size   #返回数据集大小
