#  FilesOfCsietpp

#### 简介
这是大创项目 **基于人工神经网络的曝气节能降耗系统** 的相关文件。

该项目希望建立一个ANN,降低曝气过程中的电能消耗。

#### 现状

1. 收集的数据
共收集300余条数据，输入特征为
    - ~~进水量~~
    - 出水量
    - **电单耗**
    - 进泥情况
      - 进泥量
      - 含水量
    - 加药量
    - 药单耗
    - 气水比
    - BOD（进、出水）有缺失值
    - ~~COD~~（进、出水）
    - SS（进、出水）
    - ~~NH_3-N~~（进、出水）
    - TP（进、出水）
    - PH（进、出水）
    - 氧化沟浓度
    - 回流浓度
    - ~~氧化沟出口溶解氧~~（1,2号）
    - ~~氧化沟ORP~~（1,2号）
2. 使用的模型
    1. MLPANN
        - [SimpleDataVice](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/SimpleDataVice.py)
        - [SimpleData](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/SimpleData.py)
    2. RNN
        - [RNNDemo](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/RNNForSimpleData.py)
    3. matlab中的几个模型
