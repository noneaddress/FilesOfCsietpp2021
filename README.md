#  FilesOfCsietpp

#### 简介
这是项目 **污水处理厂运行电耗调控研究**的相关文件。


该项目希望建立一个ANN,降低曝气过程中的电能消耗。
希望能预测下一天的耗电量。
#### 现状
**论文终于投出去啦！**感谢队友们推我当一作！
[论文草稿地址](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/%E6%B0%B4%E6%99%BA%E8%83%BD%E7%A0%94%E8%AE%A8%E4%BC%9A%E8%8D%89%E7%A8%BF.pdf)
<br>部分图片：
![截图](https://imgtu.com/i/q6QirV)
![截图](https://imgtu.com/i/q6QcGj)
1. 使用的模型
    1. MLPANN
        - [SimpleDataVice](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/SimpleDataVice.py)
        - [SimpleData](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/SimpleData.py)
    2. RNN
        - [RNNDemo](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/RNNForSimpleData.py)
    3. matlab中的几个模型
        1. Regression Learner自带模型 
            - Boosting
            - SVM
        2. nntool拟合的模型
        3. 自带LSTM
    [树模型与神经网络代码杂烩](https://gitee.com/nonaddress/FilesOfCsietpp/blob/master/NoteBook.py)
    4. Xgboost
        效果略差，最终论文使用此模型
    5. LightGBM
        效果最佳
2. 收集的数据
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
