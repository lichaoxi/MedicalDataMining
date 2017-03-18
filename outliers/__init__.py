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

percentage_to_remove = 20    # Remove 20% of points
print len(hw) * percentage_to_remove / 100
number_to_remove = round(len(hw) * percentage_to_remove / 100)   # 四舍五入取整
m_dist_order =  Series([float(distance.mahalanobis(hw.iloc[i], hw.mean(), np.mat(hw.cov().as_matrix()).I) ** 2)
       for i in range(len(hw))]).sort_values(ascending=False).index.tolist()

rows_to_keep_index = m_dist_order[int(number_to_remove): ]#去掉马氏距离最大的两个
my_dataframe = hw.loc[rows_to_keep_index]
print my_dataframe