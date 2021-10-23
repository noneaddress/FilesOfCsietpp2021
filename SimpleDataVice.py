import torch
import pandas as pd
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import torch.optim.lr_scheduler as scheduler
from itertools import combinations, permutations
from torch.utils.tensorboard import SummaryWriter
from torchkeras import summary, Model

'''
构建了线性层和激活层构成的神经网络
'''

# 读取原始数据
filePath = 'F:/大创/有效数据V3.csv'
initialData = pd.read_csv(filePath)
print(initialData.head(10))
print('hello world')

feaList = [i for i in range(14)]
# 选择输入特征
# todo 要不要遍历所有的输入组合
# 列举排列结果[(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)]
print(list(permutations([i for i in range(1, 4)], 2)))
# 列举组合结果[(1, 2), (1, 3), (2, 3)]
print(list(combinations([1, 2, 3], 2)))

# 分隔参数和耗电量
trainDatas = initialData.iloc[feaList, 1:301]
trainPowerConsum = initialData.iloc[-1, 1:301]
testDatas = initialData.iloc[feaList, 301:-1]
testPowerConsum = initialData.iloc[-1, 301:-1]

# 转换为dataframe
trainDatas = pd.DataFrame(trainDatas)
trainDatas = trainDatas.T

trainPowerConsum = pd.DataFrame(trainPowerConsum)

testDatas = pd.DataFrame(testDatas)
testDatas = testDatas.T

testPowerConsum = pd.DataFrame(testPowerConsum)

# 改变耗电量单位
trainPowerConsum.iloc[:, 0] = trainPowerConsum.iloc[:, 0] * 1000
testPowerConsum.iloc[:, 0] = testPowerConsum.iloc[:, 0] * 1000

# 构建dataloader
batchSize = 30
trainData = DataLoader(
    TensorDataset(
        torch.tensor(trainDatas.values).float(),
        torch.tensor(trainPowerConsum.values.astype(float)).float()
    ),
    shuffle=True, batch_size=batchSize, drop_last=True)

testData = DataLoader(
    TensorDataset(
        torch.tensor(testDatas.values.astype(float)).float(),
        torch.tensor(testPowerConsum.values.astype(float)).float()
    ),
    shuffle=False, batch_size=batchSize, drop_last=True)

# 导入待遇测数据
# predFilePath = 'F:/pred.csv'
# predData = pd.read_csv(filePath)
# d = predData.iloc[:, 10:22]
# e = predData.iloc[:, 32]
# predDataNor = pd.concat([d, e], 1)
# predDataNor = pd.DataFrame(predDataNor)
# # 构建待遇测数据的dataloader
# predDataLoader = DataLoader(TensorDataset(torch.tensor((predDataNor.values)).float()))
#

print('data is ready')

'''卷积神经网络'''


# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = torch.nn.Conv2d(25, 50, kernel_size=5)
#         self.conv2 = torch.nn.Conv2d(50, 100, kernel_size=5)
#         self.pooling = torch.nn.MaxPool2d(2)
#         self.fc = torch.nn.Linear(out_features=1)
#
#     def forward(self, x):
#         x = f.relu(self.pooling(self.conv1(x)))
#         x = f.relu(self.pooling(self.conv2(x)))
#         x


class SimpleNet(torch.nn.Module):
    def __init__(self, inputSize):
        super(SimpleNet, self).__init__()
        self.inputsize = inputSize
        self.bias = torch.nn.Parameter(torch.tensor(1, dtype=torch.float))
        self.l1 = torch.nn.Linear(self.inputsize, 14)
        self.l9 = torch.nn.Linear(14, 15)
        self.l2 = torch.nn.Linear(15, 30)
        self.l3 = torch.nn.Linear(30, 15)
        self.l4 = torch.nn.Linear(15, 5)
        self.l5 = torch.nn.Linear(5, 3)
        self.l6 = torch.nn.Linear(3, 1)
        self.l7 = torch.nn.Linear(30, 40)
        self.l8 = torch.nn.Linear(40, 30)
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, x):
        x = f.relu(self.l1(x))
        x = f.relu(self.l9(x))
        x = f.relu(self.l2(x))
        # x = f.relu(self.l3(x))
        # x = f.relu(self.l4(x))
        # x = f.relu(self.l5(x))
        # return self.l6(x)
        x = f.relu(self.l7(x))
        x = self.dropout(x)
        x = f.relu(self.l8(x))
        x = f.relu(self.l3(x))
        x = f.relu(self.l4(x))
        x = f.relu(self.l5(x))
        x = self.l6(x)
        x = x + self.bias
        return x


inputSize = trainDatas.shape[1]
learningRate = 3e-3
model = SimpleNet(inputSize=inputSize)
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

scheduler = scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # 改变学习率和动量没什么用啊
tensorboardPath = 'F:/大创/tensorboard'


# writer=SummaryWriter(tensorboardPath)
# writer.add_graph(model,input_to_model=torch.zeros(15,17))
# writer.close()
# from tensorboard import notebook
# #查看启动的tensorboard程序
# notebook.list()
def train(epoch):
    runLoss = 0.
    for batch_index, data in enumerate(trainData, 0):
        inputs, target = data
        # 输入和目标进行正则化
        inputs = (inputs - torch.min(inputs, axis=0)[0]) / (torch.max(inputs, axis=0)[0] - torch.min(inputs, axis=0)[0])
        target = (target - torch.min(target, axis=0)[0]) / (torch.max(target, axis=0)[0] - torch.min(target, axis=0)[0])
        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        # writer.add_scalar('Loss',loss.item(),epoch+1)
        runLoss += loss.item()
        print(f'{epoch + 1},{batch_index + 1},\tLoss={runLoss / 19}')
        runLoss = 0
    # writer.close()


def test(epoch):
    totalError = 0.
    model.eval()
    print('Start to test the model')
    with torch.no_grad():
        for data in testData:
            # test用于检验的数据
            # testlab 检验数据对应的标签
            test, testlab = data
            test = (test - torch.min(test, axis=0)[0]) / (torch.max(test, axis=0)[0] - torch.min(test, axis=0)[0])
            testlab = (testlab - torch.min(testlab, axis=0)[0]) / (
                        torch.max(testlab, axis=0)[0] - torch.min(testlab, axis=0)[0])

            outputs = model(test)
            # outputs=outputs*(torch.max(outputs)-torch.min(outputs))+torch.min(outputs)
            predicted = outputs.data
            # todo 反正则化，怎么实现
            testError = testlab - predicted
            # 画图
            if epoch % 50 == 0:
                xData = np.linspace(1, batchSize, batchSize)
                # if predicted.size(0) != 15:
                #     pass
                # else:
                plt.plot(xData, predicted[:, 0].numpy(), label='predicted', color='red')
                plt.plot(xData, testlab[:, 0].numpy(), label='origData', color='blue')
                plt.title(f'epoch={epoch}')
                plt.legend(loc=0)
                # plt.ylim((0, 2000))
                plt.show()

            totalError += (torch.abs(testError).sum().item())

        print(f'Average Error on test set is {totalError / 53}')


# 预测函数
# def pred():
#     for data in predDataLoader:
#         outputs = model(test)
#         _, result = outputs.data
#         # print(result)


if __name__ == '__main__':
    epochNum = 3000
    for epoch in range(epochNum):
        train(epoch)
        test(epoch)
    # pred()
