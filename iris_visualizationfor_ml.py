# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:22:10 2019

@author: u698198
"""

from sklearn.datasets import load_iris
import numpy as np
import pandas as pd 
from pandas import Series, DataFrame
from pandas.tools.plotting import andrews_curves
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
iris=load_iris()

df_iris=pd.DataFrame(iris['data'],columns=['sepallength','sepalwidth','petallength','petalwidth'])

df_iris['target']=iris['target']
print(df_iris.head())

print(df_iris['target'].value_counts())
df_iris.plot(kind='scatter',x='sepallength',y='sepalwidth')
plt.show()
sns.jointplot(x='sepallength',y='sepalwidth', data=df_iris, size=5)
sns.FacetGrid(df_iris,hue='target',size=5).map(plt.scatter,'sepallength','sepalwidth').add_legend()
sns.boxplot(x='target',y='sepallength',data=df_iris)
ax = sns.boxplot(data=df_iris, x = 'target',y = 'sepallength')
ax = sns.stripplot(data=df_iris, x='target', y='sepallength', jitter=True, edgecolor='green')
sns.violinplot(x='target',y='sepallength',data=df_iris,size=5)
sns.FacetGrid(df_iris,hue='target',size=5).map(sns.kdeplot,'sepallength').add_legend()
sns.pairplot(df_iris,hue='target',size=4)
sns.pairplot(df_iris,hue='target',size=4,diag_kind='kde')
df_iris.boxplot(by='target',figsize=(20,10))
andrews_curves(df_iris,'target')
parallel_coordinates(df_iris,'target')