import torch
import pandas as pd
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt

'''
构建简单的RNN
'''

batchSize = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netPath = 'F:/大创/RNNForSimpleData.pkl'

'''数据处理'''
# 读取原始数据
filePath = 'F:/大创/有效数据V3.csv'
initialData = pd.read_csv(filePath)
print(initialData.head(10))
print('hello world')
# 分隔参数和耗电量
feaList = [i for i in range(14)]
trainDatas = initialData.iloc[feaList, 1:301]
trainPowerConsum = pd.DataFrame(initialData.iloc[-1, 1:301]).T
trainDatas = pd.concat([trainDatas, trainPowerConsum], 0)
# trainPowerConsum与trainDatas错开一天
trainPowerConsum = initialData.iloc[-1, 2:302]
# 画图
powerConsumPlot = trainDatas.iloc[-1, :]
xData = np.linspace(1, powerConsumPlot.shape[0], 300)
plt.plot(xData, powerConsumPlot)
plt.show()

testDatas = initialData.iloc[feaList, 302:-1]
testPowerConsum = pd.DataFrame(initialData.iloc[-1, 302:-1]).T
testDatas = pd.concat([testDatas, testPowerConsum], 0)
testPowerConsum = initialData.iloc[-1, 303:]

# 转换为dataframe
trainDatas = pd.DataFrame(trainDatas)
trainDatas = trainDatas.T

trainPowerConsum = pd.DataFrame(trainPowerConsum)

testDatas = pd.DataFrame(testDatas)
testDatas = testDatas.T

testPowerConsum = pd.DataFrame(testPowerConsum)

# 改变耗电量单位
trainDatas.iloc[:, -1] = trainDatas.iloc[:, -1] * 1000
testDatas.iloc[:, -1] = testDatas.iloc[:, -1] * 1000
trainPowerConsum.iloc[:, 0] = trainPowerConsum.iloc[:, 0] * 1000
testPowerConsum.iloc[:, 0] = testPowerConsum.iloc[:, 0] * 1000

assert testPowerConsum.shape[0] == testDatas.shape[0]
assert trainDatas.shape[0] == trainPowerConsum.shape[0]

# 转换为tensor
trainDatas = torch.tensor(trainDatas.values.astype(float), device=device)
trainPowerConsum = torch.tensor(trainPowerConsum.values.astype(float), device=device)
testDatas = torch.tensor(testDatas.values.astype(float), device=device)
testPowerConsum = torch.tensor(testPowerConsum.values.astype(float), device=device)

'''不同seqlength数据构建'''

trainDatasList = list()
trainPowerConsumList = list()
for i in range(298):
    trainDatasList.append(trainDatas[i:i + 3])
    trainPowerConsumList.append(trainPowerConsum[i:i + 3])

# for i in range(300):
#     j = 300 - i
#     trainDatasList.append(trainDatas[j:])
#     trainPowerConsumList.append(trainPowerConsum[j:])
# 这里list_len==300,即batch_size==300
# seq_len==299

from torch.nn.utils.rnn import pad_sequence

# 补零
trainPowerConsum = pad_sequence(trainPowerConsumList, batch_first=True)
trainDatas = pad_sequence(trainDatasList, batch_first=True)
print(trainDatas.shape)

# 测试集batch变为1
testDatas = torch.unsqueeze(testDatas, dim=0)
testPowerConsum = torch.unsqueeze(testPowerConsum, dim=0)

'''构建dataloader'''
trainDataLoader = DataLoader(
    TensorDataset(
        trainDatas, trainPowerConsum
    ),
    shuffle=True, batch_size=batchSize, drop_last=True)

'''导入待遇测数据'''
# 测试集并不需要dataloader
# testDataLoader = DataLoader(
#     TensorDataset(
#         testDatas,testPowerConsum
#     ),
#     shuffle=False, batch_size=1)

print('Data is ready')

seqLen = 2
inputDim = len(feaList) + 1
hiddenSize = 18
numLayer = 3
learningRate = 1e-7


class RNNModel(torch.nn.Module):
    def __init__(self, inputsize, hiddensize, batchsize, numLayer):
        super(RNNModel, self).__init__()
        self.batchsize = batchsize
        self.inputsize = inputsize
        self.hiddensize = hiddensize
        self.numlayers = numLayer
        self.rnn = torch.nn.RNN(input_size=self.inputsize, hidden_size=self.hiddensize, num_layers=self.numlayers,
                                batch_first=True)
        self.l1 = torch.nn.Linear(hiddenSize, hiddensize)
        self.l2 = torch.nn.Linear(hiddenSize, 10)
        self.l3 = torch.nn.Linear(10, 8)
        self.l4 = torch.nn.Linear(8, 4)
        self.l5 = torch.nn.Linear(4, 2)
        self.l6 = torch.nn.Linear(2, 1)

    def forward(self, input, hidden):
        out, hidden = self.rnn(input.float(), hidden.float())
        batch_size, seq_len, input_dim = out.shape
        out = out.reshape(-1, input_dim)
        # out = f.sigmoid(self.l1(out))
        out = f.relu(self.l1(out))
        out = f.relu(self.l2(out))
        out = f.relu(self.l3(out))
        out = f.relu(self.l4(out))
        out = f.relu(self.l5(out))
        out = self.l6(out)
        out = out.reshape(batch_size, seq_len, -1)

        return out, hidden

    def initHidden(self):
        hidden = torch.zeros(self.numlayers, self.batchsize, self.hiddensize, device=device, dtype=torch.float64)
        return hidden


net = RNNModel(inputDim, hiddenSize, batchSize, numLayer).to(device)
criterion = torch.nn.L1Loss()  # 注意这里的损失函数
optimizer = optim.Adam(net.parameters(), lr=learningRate)


def train(epoch):
    runLoss = 0.
    optimizer.zero_grad()
    hidden = net.initHidden()

    for batchIndex, data in enumerate(trainDataLoader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs, hidden = net(inputs, hidden)
        hidden = hidden.detach()  # 注意detach位置
        loss = criterion(outputs.float(), target.float())
        # loss = loss.mean()

        loss.backward()
        optimizer.step()

    print(f'{epoch + 1},\t Loss={loss.item()}')
    return hidden
    # torch.save(net.state_dict(), netPath)


def test():
    testDatasVice = torch.clone(testDatas)
    input = testDatasVice[:, 0, :]
    input = input.view(1, 1, -1)
    assert input.shape[2] == len(feaList) + 1
    predictPowConsum = list()
    # hidden初始化，仍然用zero，没有继承上面train的hidden
    hidden = torch.zeros(numLayer, 1, hiddenSize, device=device, dtype=torch.float64)
    with torch.no_grad():
        for i in range(testDatas.shape[1]):
            output, hidden = net(input, hidden)
            if i < 51:
                testDatasVice[:, i + 1, -1] = output[0]
                input = torch.unsqueeze(testDatasVice[:, i + 1, :], dim=0)
                predictPowConsum.append(output.data.cpu().numpy().ravel()[0])
            elif i == 51:
                predictPowConsum.append(output.data.cpu().numpy().ravel()[0])
            else:
                print('\tindexError')  # 异常排除
    return predictPowConsum


def plotTest(predictPowConsum, epoch):
    predictPowConsum = torch.tensor(predictPowConsum).cpu().numpy()
    predictPowConsum = predictPowConsum * 1000
    xData = np.arange(303, 303 + testPowerConsum.size(1))
    plt.plot(xData, testPowerConsum.cpu().numpy()[0, :, 0])
    plt.plot(xData, predictPowConsum, color='red', linewidth='2')
    plt.title(f'epoch={epoch}')
    plt.show()


if __name__ == '__main__':
    epochNum = 300
    for epoch in range(epochNum):
        train(epoch)
        if epoch % 50 == 0:
            predictPowConsum = test()
            # 开始画图
            plotTest(predictPowConsum, epoch)
