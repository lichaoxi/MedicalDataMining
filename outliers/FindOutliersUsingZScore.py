# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from numpy import float64
#身高数据
Height_cm = np.array([164, 167, 168, 169, 169, 170, 170, 170, 171, 172, 172, 173, 173, 175, 176, 178], dtype=float64)
#体重数据
Weight_kg = np.array([54,  57,  58,  60,  61,  60,  61,  62,  62,  64,  62,  62,  64,  56,  66,  70], dtype=float64)
hw = {'Height_cm': Height_cm, 'Weight_kg': Weight_kg}
hw = pd.DataFrame(hw)#hw为矩阵 两列两个变量（身高和体重） 行为变量序号
print hw


from sklearn import preprocessing
from matplotlib import pyplot as plt
#scale 数据进行标准化    公式为：(X-mean)/std  将数据按按列减去其均值，并处以其方差   结果是所有数据都聚集在0附近，方差为1。
is_height_outlier = abs(preprocessing.scale(hw['Height_cm'])) > 2 #线性归一化    数据中心的标准偏差与2比较(建议用3比较)
is_weight_outlier = abs(preprocessing.scale(hw['Weight_kg'])) > 2
is_outlier = is_height_outlier | is_weight_outlier#按位或 表示两个变量中有一位是异常的 本组数据（体重，身高）异常    is_outlier是数组，值为True或False
color = ['g', 'r']
pch = [1 if is_outlier[i] == True else 0 for i in range(len(is_outlier))]#pch是数组，值为1或0,1表示异常点
cValue = [color[is_outlier[i]] for i in range(len(is_outlier))]#颜色数组
# print is_height_outlier
# print cValue
fig = plt.figure()
plt.title('Scatter Plot')
plt.xlabel('Height_cm')
plt.ylabel('Weight_kg')
plt.scatter(hw['Height_cm'], hw['Weight_kg'], s=40, c=cValue)#散点图 s代表图上点大小
plt.show()