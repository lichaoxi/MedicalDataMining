# -*- coding: utf-8 -*-
#其他变量不变，对一个变量进行异常检测，无变量相关性影响
import numpy as np
import pandas as pd
df = pd.read_csv("E:\Workspaces\MyEclipse\MedicalDataMining/outliers/farequote.csv")

#print df.head()
#print np.unique(df.airline)#航空公司

dd = df.query('airline=="AAL"') ## 得到法航的数据KLM

#print dd.responsetime.describe()#个数，均值，方差，最大最小值

#基于标准差得异常检测    95.449974面积在平均数左右两个标准差的范围内
# def a1(dataframe, threshold=.95):
#     d = dataframe['responsetime']
#     dataframe['isAnomaly'] = d > d.quantile(threshold)  
#     return dataframe
# #print a1(dd)
#  
# #基于ZSCORE的异常检测
# def a2(dataframe, threshold=3.5):
#     d = dataframe['responsetime']
#     zscore = (d - d.mean())/d.std()
#     dataframe['isAnomaly'] = zscore.abs() > threshold
#     return dataframe
# #增强的zscore
# def a3(dataframe, threshold=3.5):
#     dd = dataframe['responsetime']
#     MAD = (dd - dd.median()).abs().median()
#     zscore = ((dd - dd.median())* 0.6475 /MAD).abs()
#     dataframe['isAnomaly'] = zscore > threshold
#     return dataframe
#  
# #数据可视化
#  
# import matplotlib.pyplot as plt
# da = a1(dd)
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)#一块画布多个图
# ax2 = fig.add_subplot(2, 1, 2)
# ax1.plot(da['responsetime'])
# ax2.plot(da['isAnomaly'])
# plt.show()


import matplotlib.pyplot as plt
from matplotlib import pyplot  
from sklearn import svm  
from sklearn.cluster   import KMeans 
from scipy import stats 
#基于KMEAN的聚集算法
def a4(dataframe, threshold = .9):
    ## add one dimention of previous response
    #前一个点的响应时间增加到当前点，第一个点的该值为0，命名该列为t0
    preresponse = 0
    newcol = []
    newcol.append(0)#第一个点的该值为0
    for index, row in dataframe.iterrows():
        if preresponse != 0:
            newcol.append(preresponse)
        preresponse = row.responsetime
    dataframe["t0"] = newcol
    
#     plt.scatter(dataframe.t0,dataframe.responsetime)#<t0,time>画出散点图
#     plt.show()
    
    ## remove first row as there is no previous event for time
    dd = dataframe.drop(dataframe.head(1).index) 
    clf = KMeans(n_clusters=2)
    X=np.array(dd[['responsetime','t0']])
    cls = clf.fit_predict(X)
    freq = stats.itemfreq(cls)
    (A,B) = (freq[0,1],freq[1,1])
    t = abs(A-B)/float(max(A,B))
    if t > threshold :
        ## "Anomaly Detected!"
        index = freq[0,0]
        if A > B :
            index = freq[1,0]
        dd['isAnomaly'] = (cls == index)
    else :
        ## "No Anomaly Point"
        dd['isAnomaly'] = False
    return dd

da = a4(dd)
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)#一块画布多个图
ax2 = fig.add_subplot(2, 1, 2)
ax1.plot(da['responsetime'])
ax2.plot(da['isAnomaly'])
plt.show()
