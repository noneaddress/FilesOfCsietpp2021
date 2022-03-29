trainDatasFour = initialDataCopyForDropNormal.iloc[1:301, feaListFour]
trainPowerConsumFour = initialDataCopyForDropNormal.iloc[1:301, -1]
testDatasFour = initialDataCopyForDropNormal.iloc[302:-1, feaListFour]
testPowerConsumFour = initialDataCopyForDropNormal.iloc[302:-1, -1]
trainPowerConsumFour = pd.DataFrame(trainPowerConsumFour)
testPowerConsumFour = pd.DataFrame(testPowerConsumFour)