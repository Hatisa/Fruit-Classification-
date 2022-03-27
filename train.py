from getdata import FRDataset as FRD
from torch.utils.data import DataLoader as DataLoader
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn

dataset_dir = './data/'

model_cp = './model/'               # 网络参数保存位置
workers = 5                        # PyTorch读取数据线程数量
batch_size = 128                     # batch_size大小
lr = 0.01                         # 学习率
nepoch = 30

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
# .to(device)

def train():
    datafile = FRD('train', dataset_dir)              # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)     # 用PyTorch的DataLoader类封装，实现数据集顺序打乱，多线程读取，一次取多个数据等效果

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    model = Net()                       # 实例化一个网络
    model = model.to(device)               # 网络送入GPU，即采用GPU计算，如果没有GPU加速，可以去掉".cuda()"
    # model = model
    # model = nn.DataParallel(model)
    model.train()                       # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)         # 实例化一个优化器，即调整网络参数，优化方式为adam方法

    criterion = torch.nn.CrossEntropyLoss()                         # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小


    cnt = 0             # 训练图片数量
    for epoch in range(nepoch):
        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为128，
        # 所以每次读取的数据量为128，即img包含了128个图像，label有128个
        for img, label in dataloader:
            img, label = Variable(img).to(device), Variable(label).to(device)
            out = model(img)
            loss = criterion(out, label.squeeze())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            cnt += 1

            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt*batch_size, loss/batch_size))
            # 打印一个batch size的训练结果

    torch.save(model.state_dict(), '{0}/model.pth'.format(model_cp))            # 训练所有数据后，保存网络的参数


if __name__ == '__main__':
    train()