# %%
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from matplotlib import pyplot as plt
import torch.optim.lr_scheduler as scheduler

import seaborn as sns
from sklearn.metrics import r2_score

#%%
# 读取原始数据
filePath = 'F:/大创/有效数据改v2.csv'
initialData = pd.read_csv(filePath)
print(initialData.iloc[:, 1])
print('hello world')

# %%
'''绘制热力图'''
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 设置显示中文 需要先安装字体 aistudio
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# %%
from sklearn.preprocessing import MinMaxScaler,StandardScaler
scaler=MinMaxScaler()
scaler = StandardScaler()
initialDataVice = scaler.fit_transform(initialDataVice)
# TestData = scaler.fit_transform()
title = initialDataVice.columns
# temp = initialData.iloc[np.r_[0:7, 8], 1:]

temp = initialDataVice
temp.columns = title
#%%
temp2 = initialData.corr(method='pearson')
fig, ax = plt.subplots(figsize=(20, 20), dpi=100)
plt.xticks(size=20)
plt.yticks(size=20)
sns.heatmap(temp2, annot=True, square=True, cmap='Reds', linewidths=.5)
# 设置colorbar的刻度字体大小
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=10)
# 设置colorbar的label文本和字体大小
fontDic = {'family': 'times new roman',
           'color': 'k',
           'weight': 'normal',
           'size': 20, }
cbar = ax.collections[0].colorbar
cbar.set_label('Pearson Correlation Coefficient', fontdict=fontDic)
plt.show()

# %%
# 分隔参数和耗电量
feaList = [i for i in range(7)]
# a list records which feature is selected for trainning
trainDatas = initialData.iloc[feaList, 1:301]
trainPowerConsum = initialData.iloc[-1, 1:301]
testDatas = initialData.iloc[feaList, 302:-1]
testPowerConsum = initialData.iloc[-1, 302:-1]
# %%
# 转换为dataframe
trainDatas = pd.DataFrame(trainDatas)
trainDatas = trainDatas.T
trainDatas.columns = initialData.iloc[feaList, 0]
trainPowerConsum = pd.DataFrame(trainPowerConsum)

testDatas = pd.DataFrame(testDatas)
testDatas = testDatas.T
testDatas.columns = initialData.iloc[feaList, 0]
testPowerConsum = pd.DataFrame(testPowerConsum)
# %%
trainPowerConsum.iloc[:, 0] = trainPowerConsum.iloc[:, 0] * 1000
testPowerConsum.iloc[:, 0] = testPowerConsum.iloc[:, 0] * 1000
# %%

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


# %%
class SimpleNet(torch.nn.Module):
    def __init__(self, inputSize):
        super(SimpleNet, self).__init__()
        self.inputsize = inputSize
        # self.bias = torch.nn.Parameter(torch.tensor(1, dtype=torch.float))
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
        # x = x + self.bias
        return x


# %%
inputSize = trainDatasFour.shape[1]
learningRate = 5e-3
model = SimpleNet(inputSize=inputSize)
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=learningRate)

scheduler = scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)  # 改变学习率和动量没什么用啊


# %%
def train(epoch):
    runLoss = 0.
    for batch_index, data in enumerate(trainData, 0):
        inputs, target = data
        # 输入和目标进行正则化
        inputs = (inputs - torch.min(inputs, axis=0)[0]) / (torch.max(inputs, axis=0)[0] - torch.min(inputs, axis=0)[0])
        # target = (target - torch.min(target, axis=0)[0]) / (torch.max(target, axis=0)[0] - torch.min(target, axis=0)[0])
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
# %%
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
            # testlab = (testlab - torch.min(testlab, axis=0)[0]) / (                    torch.max(testlab, axis=0)[0] - torch.min(testlab, axis=0)[0])

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

            totalError = r2_score(testlab, predicted)

            print(f'RSquare Error on test set is {totalError}')

# %%
# 预测函数
# def pred():
#     for data in predDataLoader:
#         outputs = model(test)
#         _, result = outputs.data
#         # print(result)

# %%
epochNum = 500
for epoch in range(epochNum):
    train(epoch)
    test(epoch)

# # %%
# # GBDT
# xData = np.linspace(1, temp.shape[0], temp.shape[0])
# plt.plot(xData, temp, label='predicted', color='red')
# plt.plot(xData, trainPowerConsum, label='origData', color='blue')
# plt.legend(loc=0)
# plt.show()
#
# # %%
#
# import lightgbm as lgb
# from sklearn.model_selection import KFold
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
#
# # %%
#
# CommonDataPath = 'F:/新建文件夹/Mathorcup/MyCode/'
# # PrinComPredData = trainDatas
# PrinComTrainData = trainDatas
# TrainDataClearPrice = trainPowerConsum.astype('float')
# scaler = StandardScaler()
# PrinComTrainData = scaler.fit_transform(PrinComTrainData)
# # TestData = scaler.fit_transform()
#
# # %%
# params = {'learning_rate': 0.001,
#           'boosting_type': 'gbdt',
#           'objective': 'regression_l1',
#           'metric': 'mae',
#           'min_child_samples': 30,
#           'min_child_weight': 0.01,
#           'feature_fraction': 0.8,
#           'bagging_fraction': 0.8,
#           'bagging_freq': 2,
#           'num_leaves': 32,
#           'max_depth': 8,
#           'n_jobs': -1,
#           'seed': 2019,
#           'verbose': -1,
#           }
#
#
# # %%
# def ModelAcc(rawPrice, predPrice):
#     """
#     参数
#     rawPrice -- 实际值
#     y_pred -- 预测值
#
#     最后返回题目给出的评价指标
#     """
#     n = len(rawPrice)
#     Mape = sum(np.abs((rawPrice - predPrice) / rawPrice)) / n
#     ApeLess005 = pd.DataFrame(abs(rawPrice - predPrice) / rawPrice)
#     Acc = 0.2 * (1 - Mape) + (ApeLess005[ApeLess005 <= 0.05].count() /
#                               ApeLess005.count()) * 0.8
#     return Acc
#
# # %%
#
# folds = 7
# valPred = np.zeros(len(PrinComTrainData))
# valTrue = np.zeros(len(PrinComTrainData))
# # preds = np.zeros(len(TestData))
# kfold = KFold(n_splits=folds, shuffle=True, random_state=4321)
# for fold, (trnInd, valInd) in enumerate(kfold.split(PrinComTrainData, TrainDataClearPrice)):
#     print('fold ', fold + 1)
#     xTrn, yTrn, xVal, yVal = PrinComTrainData[trnInd], TrainDataClearPrice.iloc[trnInd, 0], PrinComTrainData[valInd], \
#                              TrainDataClearPrice.iloc[valInd, 0]
#     trainSet = lgb.Dataset(xTrn, yTrn)
#     validSet = lgb.Dataset(xVal, yVal)
#
#     model = lgb.train(params, trainSet, num_boost_round=5000, valid_sets=(trainSet, validSet),
#                       early_stopping_rounds=500, verbose_eval=False)
#     valPred[valInd] += model.predict(xVal, predict_disable_shape_check=True)
#     # preds += model.predict(testDatas, predict_disable_shape_check=True) / folds
#     valTrue[valInd] += yVal
#
# # %%
#
# print('Accuracy ', round(ModelAcc(valTrue, valPred), 7))
#
# # %%
# preds = np.zeros(len(testDatas))
# preds += model.predict(testDatas, predict_disable_shape_check=True)

# %%
import xgboost as xgb

# https://zhuanlan.zhihu.com/p/33700459
# https://zhuanlan.zhihu.com/p/83620830
xg_reg = xgb.XGBRegressor(
    objective='reg:linear',
    colsample_bytree=0.45,
    learning_rate=0.01,
    max_depth=5,
    n_estimators=16,
    alpha=0,
    gamma=0, subsample=0.7
)
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(trainDatas, trainPowerConsum.astype('float'), test_size=0.3)
xg_reg.fit(xTrain, yTrain)

# %%

from matplotlib import rc

rc('font', family='Times New Roman')
rc('text', usetex=True)


# %%
def validModel(predict_x, initial, MyModel):
    temp = MyModel.predict(predict_x)
    xData = np.linspace(1, temp.shape[0], temp.shape[0])

    fig, ax = plt.subplots(dpi=150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(xData, temp, label='predicted', color='red')
    plt.plot(xData, initial, label='origData', color='blue')
    font_dict = dict(fontsize=16,
                     color='black',
                     family='Times New Roman',
                     weight='light',
                     style='italic',
                     )
    fontdict_prop = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 23,
                     }
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('Time/Day', fontdict=font_dict)
    plt.ylabel(r"Power Consumption Kw·h per cubic meters", fontdict=font_dict)
    plt.legend(loc=0, frameon=False, prop=fontdict_prop)
    plt.title('R Square is {0:.3f}'.format(r2_score(initial, temp)), fontdict=font_dict)
    plt.show()
    print(r2_score(initial, temp))


# %%
import pickle

# 关于保存模型的文章
# https://zhuanlan.zhihu.com/p/370761158
# https://zhuanlan.zhihu.com/p/99173424
# 路径里不要有中文名称，特别是读取模型的时候
# 保存文件
xg_reg.save_model('E:/DEVSideCar/XGBModelv5.json')
# json格式文件
# 读取文件
# 法一
# 先实例化
# model_xgb_2 = xgb.Booster()
# model_xgb_2.load_model("model.json")
# 法二
# clf = XGBClassifier()
# booster = Booster()
# booster.load_model('./model.xgb')
# clf._Booster = booster
# %%
plt.subplots_adjust(left=0.8, top=0.9, bottom=0.1)
plt.figure(figsize=(25, 25), dpi=150)
plt.xticks(size=40)
xgb.plot_importance()
plt.savefig('F:/大创/XgRegSelected6FeaVice.png', dpi=150, bbox_inches='tight')

plt.show()

# %%

filePathVice = 'F:/QQ/包含电单耗共15个参数353日有效数据.xlsx'
initialDataVice = pd.read_excel(filePathVice)
print(initialDataVice.head(5))

# %%
initialDataVice.drop(labels=initialDataVice.columns[np.r_[0, -1]], axis=1, inplace=True)
# %%
# 分隔参数和耗电量
feaListVice = [i for i in range(0, 14)]
# a list records which feature is selected for trainning
trainDatasVice = initialDataVice.iloc[1:301, feaListVice]
trainPowerConsumVice = initialDataVice.iloc[1:301, -1]
testDatasVice = initialDataVice.iloc[302:-1, feaListVice]
testPowerConsumVice = initialDataVice.iloc[302:-1, -1]
trainPowerConsumVice = pd.DataFrame(trainPowerConsumVice)
testPowerConsumVice = pd.DataFrame(testPowerConsumVice)
# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
trainDatasVice = scaler.fit_transform(trainDatasVice)
trainPowerConsumVice = scaler.fit_transform(trainPowerConsumVice)

# %%
import xgboost as xgb

# https://zhuanlan.zhihu.com/p/33700459
XgReg14Fea = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.6,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=20,
    alpha=0,
    gamma=0,
    subsample=0.9
)
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(trainDatasVice, trainPowerConsumVice.astype('float'), test_size=0.3)
XgReg14Fea.fit(xTrain, yTrain)

# %%

validModel(testDatasVice, testPowerConsumVice, XgReg14Fea)
validModel(trainDatasVice, trainPowerConsumVice, XgReg14Fea)

# %%
XgReg14Fea.save_model('E:/DEVSideCar/XGBModel14FeaV3.json')

# %%

'''筛选变量后的数据'''

initialDataThd = initialDataVice.copy(deep=True)
feaListThd = [0, 1, 7, 8, 11, 13]
# a list records which feature is selected for trainning
trainDatasThd = initialDataThd.iloc[1:301, feaListThd]
trainPowerConsumThd = initialDataThd.iloc[1:301, -1]
testDatasThd = initialDataThd.iloc[302:-1, feaListThd]
testPowerConsumThd = initialDataThd.iloc[302:-1, -1]
trainPowerConsumThd = pd.DataFrame(trainPowerConsumThd)
testPowerConsumThd = pd.DataFrame(testPowerConsumThd)

# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split

# https://zhuanlan.zhihu.com/p/33700459
XgRegSelected6Fea = xgb.XGBRegressor(
    objective='reg:squarederror',
    colsample_bytree=0.6,
    learning_rate=0.06,
    max_depth=6,
    n_estimators=21,
    alpha=0,
    gamma=0,
    subsample=0.9,
)

xTrain, xTest, yTrain, yTest = train_test_split(trainDatasThd, trainPowerConsumThd.astype('float'), test_size=0.3)
XgRegSelected6Fea.fit(xTrain, yTrain)
# %%
validModel(testDatasThd, testPowerConsumThd, XgRegSelected6Fea)
validModel(trainDatasThd, trainPowerConsumThd, XgRegSelected6Fea)
# %%

# XgRegSelected6FeaVice = xgb.XGBRegressor()
# XgRegSelected6FeaVice.load_model('E:/DEVSideCar/XGBModelSelected6FeaV8.json')
# validModel(testDatasThd, testPowerConsumThd, XgRegSelected6FeaVice)
# validModel(trainDatasThd, trainPowerConsumThd, XgRegSelected6FeaVice)
#
# # %%
# XgReg7Fea = xgb.XGBRegressor()
# XgReg7Fea.load_model('E:/DEVSideCar/XGBModelv5.json')
# validModel(testDatas, testPowerConsum, XgReg7Fea)
# validModel(trainDatas, trainPowerConsum, XgReg7Fea)
#
# # %%
# XgReg14Fea = xgb.XGBRegressor()
# XgReg14Fea.load_model('E:/DEVSideCar/XGBModel14FeaV3.json')
# validModel(testDatasVice, testPowerConsumVice, XgReg14Fea)
# validModel(trainDatasVice, trainPowerConsumVice, XgReg14Fea)


# %%
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

XgRegSelected6Fea = xgb.XGBRegressor(
    objective='reg:squarederror',
    subsample=0.8,
    alpha=0,
    gamma=0
)
parameters = {
    'max_depth': [3, 4, 5, 6, 7],
    'n_estimators': [i for i in range(15, 23)],
    'learning_rate': [i for i in np.arange(0.01, 0.15, 0.05)],
    'colsample_bytree': [i for i in np.arange(0.4, 0.9, 0.3)]
}
from sklearn.metrics import accuracy_score, make_scorer

scoreDict = {
    'adj': 'r2',
    'b': 'max_error'
}
gs = GridSearchCV(estimator=XgRegSelected6Fea, param_grid=parameters, cv=5, verbose=3, refit='adj', scoring=scoreDict, n_jobs=-1)
gs.fit(trainDatasThd, trainPowerConsumThd)
bestReg = gs.best_estimator_

# %%
myReg = xgb.XGBRegressor()
tempPath = 'E:\\DEVSideCar\\XGBModelSelected6FeaV1WithoutGridSearch.json'
myReg.load_model(tempPath)
validModel(testDatasThd, testPowerConsumThd, myReg)
validModel(trainDatasThd, trainPowerConsumThd, myReg)
# %%

myReg2 = xgb.XGBRegressor()
myReg2.load_model('E:/DEVSideCar/XGBModel14FeaV3.json')

# %%

plt.subplots_adjust(left=0.8, top=0.9, bottom=0.1)
plt.figure(figsize=(25, 25), dpi=150)
plt.xticks(size=40)
xgb.plot_importance(myReg)
plt.savefig('F:/大创/XgRegSelected6FeaVice.png', dpi=150, bbox_inches='tight')
plt.show()

# %%
feaListFour = [0, 1, 4, 7, 8, 10, 11, 13]
initialDataFour = initialDataVice.copy(deep=True)
#%%
# 正态分布
# 3sigma准则 --->
#  mean() - 3* std() ---下限
#  mean() + 3* std() ---上限
# 自实现3sigma 原则
def three_sigma(ser):
    """
    自实现3sigma 原则
    :param ser: 数据
    :return: 处理完成的数据
    """
    bool_id = ((ser.mean() - 3 * ser.std()) <= ser) & (ser <= (ser.mean() + 3 * ser.std()))
    # bool数组索引
    # ser[bool_id]
    return ser.index[bool_id]

#%%
abnormalIndex=three_sigma(pd.Series(initialDataFour.iloc[:,-1]))
#%%

trainDatasFour = initialDataFour.iloc[1:301, feaListFour]
trainPowerConsumFour = initialDataFour.iloc[1:301, -1]
testDatasFour = initialDataFour.iloc[302:-1, feaListFour]
testPowerConsumFour = initialDataFour.iloc[302:-1, -1]
trainPowerConsumFour = pd.DataFrame(trainPowerConsumFour)
testPowerConsumFour = pd.DataFrame(testPowerConsumFour)

# %%
trainDatasFour = initialDataCopyForDropNormal.iloc[1:301, feaListFour]
trainPowerConsumFour = initialDataCopyForDropNormal.iloc[1:301, -1]
testDatasFour = initialDataCopyForDropNormal.iloc[302:-1, feaListFour]
testPowerConsumFour = initialDataCopyForDropNormal.iloc[302:-1, -1]
trainPowerConsumFour = pd.DataFrame(trainPowerConsumFour)
testPowerConsumFour = pd.DataFrame(testPowerConsumFour)
#%%
import xgboost as xgb
from sklearn.model_selection import train_test_split

# https://zhuanlan.zhihu.com/p/33700459
XgRegSelected8Fea = xgb.XGBRegressor(
    objective='reg:squarederror',
    booster='gbtree',
    colsample_bytree=0.6,
    learning_rate=0.06,
    max_depth=3,
    n_estimators=124,
    alpha=0,
    gamma=0,
    subsample=0.8,
    verbosity=2,
    num_leaves=5,
    # min_child_weight=0.001
)
XgRegSelected8Fea.fit(xTrain, yTrain)
# %%
from sklearn.model_selection import GridSearchCV
myParameterDict={
    # 'max_depth':[i for i in range(3,20)],
    # 'num_leaves':[i for i in range(5,100,5)]
    'min_chlid_weight':[i for i in np.arange(0.001,1,0.002)]
    # 'n_estimators':[i for i in range(50,300)],
    # 'learning_rate':[i for i in np.arange(0.001,0.02,0.001)],
    # 'colsamp_bytree':[i for i in np.arange(0.3,1,0.1)],
    # 'gamma':[i for i in np.arange(0.3,0.8,0.1)],
    # 'alpha':[i for i in np.arange(0.3,0.8,0.1)]
}

scoreDict={
    'r2':'r2'
}
GsSelected8Fea=GridSearchCV(estimator=XgRegSelected8Fea,param_grid=myParameterDict,cv=5,verbose=3,refit='r2',scoring=scoreDict,n_jobs=-1)
xTrain, xTest, yTrain, yTest = train_test_split(trainDatasFour, trainPowerConsumFour.astype('float'), test_size=0.3)
GsSelected8Fea.fit(xTrain,yTrain)
bestRegSelected8Fea=GsSelected8Fea.best_estimator_
#%%
validModel(testDatasFour, testPowerConsumFour, XgRegSelected8Fea)
validModel(trainDatasFour, trainPowerConsumFour, XgRegSelected8Fea)
# %%

plt.subplots_adjust(left=0.8, top=0.9, bottom=0.1)
plt.figure(figsize=(25, 25), dpi=150)
plt.xticks(size=40)
xgb.plot_importance(XgRegSelected8Fea)
plt.savefig('F:/大创/XgRegSelected8Fea.png', dpi=150, bbox_inches='tight')
plt.show()
# %%
params = {
    'booster': 'gbtree',
    'objective': 'reg:squarederror',
    'colsample_bytree': 0.6,
    'learning_rate': 0.06,
    'max_depth': 6,
    'n_estimators': 22,
    'alpha': 0,
    'gamma': 0,
    'subsample': 0.9
}

dataForTrain = xgb.DMatrix(trainDatasFour, trainPowerConsumFour)
num_rounds = 400
plst = params.items()
myTempModel = xgb.train(list(plst), dataForTrain, num_rounds)

# %%

import lightgbm as lgb
lgbReg=lgb.LGBMRegressor()
lgbReg.fit(xTrain,yTrain,eval_set=[(xTest,yTest)])

#%%
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'l2',
    'nthread': 4,
    'learning_rate': 0.1,
    'num_leaves': 30,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimator':1000
}

data_train = lgb.Dataset(xTrain, yTrain)
cv_results = lgb.cv(params, data_train, num_boost_round=1000, nfold=5, shuffle=True, metrics='l2',)
#%%
print('bestn_estimators:', len(cv_results['auc-mean']))
print('bestcvscore:', pd.Series(cv_results['auc-mean']).max())

#%%
# %%

# 构建dataloader
batchSize = 30
trainData = DataLoader(
    TensorDataset(
        torch.tensor(trainDatasFour.values).float(),
        torch.tensor(trainPowerConsumFour.values.astype(float)).float()
    ),
    shuffle=True, batch_size=batchSize, drop_last=True)

testData = DataLoader(
    TensorDataset(
        torch.tensor(testDatasFour.values.astype(float)).float(),
        torch.tensor(testPowerConsumFour.values.astype(float)).float()
    ),
    shuffle=False, batch_size=batchSize, drop_last=True)

#%%
def plotNN(batchSize,predicted,testlab):
    xData = np.linspace(1, batchSize, batchSize)
                # if predicted.size(0) != 15:
                #     pass
                # else:
    plt.plot(xData, predicted[:, 0], label='predicted', color='red')
    plt.plot(xData, testlab[:, 0], label='origData', color='blue')
    # plt.title(f'epoch={epoch}')
    plt.legend(loc=0)
    # plt.ylim((0, 2000))
    plt.show()

#%%
def plotNNIO(predict, initial):
    xData = np.linspace(1, temp.shape[0], temp.shape[0])

    fig, ax = plt.subplots(dpi=150)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.plot(xData, predict, label='predicted', color='red')
    plt.plot(xData, initial, label='origData', color='blue')
    font_dict = dict(fontsize=16,
                     color='black',
                     family='Times New Roman',
                     weight='light',
                     style='italic',
                     )
    fontdict_prop = {'family': 'Times New Roman',
                     'weight': 'normal',
                     'size': 23,
                     }
    plt.xticks(fontproperties='Times New Roman', size=14)
    plt.yticks(fontproperties='Times New Roman', size=14)
    plt.xlabel('Time/Day', fontdict=font_dict)
    plt.ylabel(r"Power Consumption Kw·h per cubic meters", fontdict=font_dict)
    plt.legend(loc=0, frameon=False, prop=fontdict_prop)
    plt.title('R Square is {0:.3f}'.format(r2_score(initial, temp)), fontdict=font_dict)
    plt.show()
    print(r2_score(initial, temp))
