# -*- coding: utf-8 -*-
#马氏距离用于非线性数据时，出现异常
#马氏距离检查异常值,值越大，该点为异常点可能性越大
import pandas as pd
from sklearn import preprocessing
import numpy as np
from numpy import float64
from matplotlib import pyplot as plt
from scipy.spatial import distance
from pandas import Series

x = np.array([4,  8, 10, 16, 17, 22, 27, 33, 38, 40, 47, 48, 53, 55, 63, 71, 76, 85, 85, 92, 96], dtype=float64)
y = np.array([6, 22, 32, 34, 42, 51, 59, 63, 64, 69, 70, 20, 70, 63, 63, 55, 46, 41, 33, 19,  6], dtype=float64)
hw = {'x': x, 'y': y}
hw = pd.DataFrame(hw)

percentage_of_outliers = 10    # Mark 10% of points as outliers
number_of_outliers = round(len(hw) * percentage_of_outliers / 100)   # 四舍五入取整
m_dist_order =  Series([float(distance.mahalanobis(hw.iloc[i], hw.mean(), np.mat(hw.cov().as_matrix()).I) ** 2)
       for i in range(len(hw))]).sort_values(ascending=False).index.tolist()

rows_not_outliers = m_dist_order[int(number_of_outliers): ]
my_dataframe = hw.loc[rows_not_outliers]

is_outlier = [True, ] * 21
for i in rows_not_outliers:
    is_outlier[i] = False
color = ['g', 'r']
pch = [1 if is_outlier[i] == True else 0 for i in range(len(is_outlier))]
cValue = [color[is_outlier[i]] for i in range(len(is_outlier))]
fig = plt.figure()
plt.title('Scatter Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(hw['x'], hw['y'], s=40, c=cValue)
plt.show()
