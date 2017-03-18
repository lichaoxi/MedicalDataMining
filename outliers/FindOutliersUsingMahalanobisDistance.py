# -*- coding: utf-8 -*-
#马氏距离检查异常值,值越大，该点为异常点可能性越大
import pandas as pd
import numpy as np
from numpy import float64
from matplotlib import pyplot as plt
from scipy.spatial import distance
from pandas import Series
from mpl_toolkits.mplot3d  import Axes3D
#身高数据
Height_cm = np.array([164, 167, 168, 168, 169, 169, 169, 170, 172, 173, 175, 176, 178], dtype=float64)
#体重数据
Weight_kg = np.array([55,  57,  58,  56,  57,  61,  61,  61,  64,  62,  56,  66,  70], dtype=float64)
#年龄数据
Age = np.array([13,  12,  14,  17,  15,  14,  16,  16,  13,  15,  16,  14,  16], dtype=float64)
hw = {'Height_cm': Height_cm, 'Weight_kg': Weight_kg, 'Age': Age}#hw为矩阵 三列三个变量（身高、体重、年龄） 行为变量序号
hw = pd.DataFrame(hw)
print len(hw)#13

n_outliers = 2#选2个作为异常点
#iloc[]取出3列，一行    hw.mean()此处为3个变量的数组    np.mat(hw.cov().as_matrix()).I为协方差的逆矩阵    **为乘方
#Series的表现形式为：索引在左边，值在右边
#m_dist_order为一维数组    保存Series中值降序排列的索引
m_dist_order =  Series([float(distance.mahalanobis(hw.iloc[i], hw.mean(), np.mat(hw.cov().as_matrix()).I) ** 2)
       for i in range(len(hw))]).sort_values(ascending=False).index.tolist()
is_outlier = [False, ] * 13
for i in range(n_outliers):#马氏距离值大的标为True
    is_outlier[m_dist_order[i]] = True
# print is_outlier

color = ['g', 'r']
pch = [1 if is_outlier[i] == True else 0 for i in range(len(is_outlier))]
cValue = [color[is_outlier[i]] for i in range(len(is_outlier))]
# print cValue

fig = plt.figure()
#ax1 = fig.add_subplot(111, projection='3d')
ax1 = fig.gca(projection='3d')
ax1.set_title('Scatter Plot')
ax1.set_xlabel('Height_cm')
ax1.set_ylabel('Weight_kg')
ax1.set_zlabel('Age')
ax1.scatter(hw['Height_cm'], hw['Weight_kg'], hw['Age'],  s=40, c=cValue)
plt.show()