from getdata import FRDataset as FRD
from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata

dataset_dir = './data/test/'                    # 数据集路径
model_file = './model/model.pth'                # 模型保存路径
N = 10

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# new version
def test():

    # setting model
    model = Net()                                       # 实例化一个网络
    model.to(device)                                       # 送入GPU，利用GPU计算
    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    # get data
    files = random.sample(os.listdir(dataset_dir), N)   # 随机获取N个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    for file in files:
        img = Image.open(dataset_dir + file)            # 打开图像
        img_data = getdata.dataTransform(img).to(device)           # 转换成torch tensor数据   ++

        imgs.append(img)                                # 图像list
        imgs_data.append(img_data)                      # tensor list
    imgs_data = torch.stack(imgs_data)                  # tensor list合成一个4D tensor

    # calculation
    out = model(imgs_data)                              # 对每个图像进行网络计算
    out = F.softmax(out, dim=1)                         # 输出概率化
    out = out.data.cpu().numpy()                        # 转成numpy数据

    # pring results         显示结果
    for idx in range(N):
        plt.figure()
        A= out[idx, 0]
        for i in range(32):
            if out[idx, i+1]>A:
                A= out[idx, i+1]
                fruit= i+1

        if fruit==0:
            result='Apple Braeburn'
        elif fruit==1:
            result='Apple Granny Smith'
        elif fruit==2:
            result='Apricot'
        elif fruit==3:
            result='Avocado'
        elif fruit==4:
            result='Banana'
        elif fruit==5:
            result='Blueberry'
        elif fruit==6:
            result='Cactus fruit'
        elif fruit==7:
            result='Cantaloupe'
        elif fruit==8:
            result='Cherry'
        elif fruit==9:
            result='Clementine'
        elif fruit==10:
            result='Corn'
        elif fruit==11:
            result='Cucumber Ripe'
        elif fruit==12:
            result='Grape Blue'
        elif fruit==13:
            result='Kiwi'
        elif fruit==14:
            result='Lemon'
        elif fruit==15:
            result='Limes'
        elif fruit==16:
            result='Mango'
        elif fruit==17:
            result='Onion White'
        elif fruit==18:
            result='Orange'
        elif fruit==19:
            result='Papaya'
        elif fruit==20:
            result='Passion Fruit'
        elif fruit==21:
            result='Peach'
        elif fruit==22:
            result='Pear'
        elif fruit==23:
            result='Pepper Green'
        elif fruit==24:
            result='Pepper Red'
        elif fruit==25:
            result='Pineapple'
        elif fruit==26:
            result='Plum'
        elif fruit==27:
            result='Pomegranate'
        elif fruit==28:
            result='Potato Red'
        elif fruit==29:
            result='Raspberry'
        elif fruit==30:
            result='Strawberry'
        elif fruit==31:
            result='Tomato'
        else:
            result='Watermelon'


        plt.suptitle('{} : {:.2%}'.format(result,out[idx, fruit]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()


