# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 09:31:25 2020

@author: Shrita
"""

import pandas as pd
import pickle


df1 = pd.read_csv('C:\\Users\\Shrita\\Desktop\\Career-recomendation-system\\Codes\\ad_pr.csv')
df1 = df1.drop('Serial No.',axis = 1)
X10 = df1.drop(['Chance of Admit'], axis=1)
y10 = df1['Chance of Admit']
from sklearn.model_selection import train_test_split
X_train10, X_test10, y_train10, y_test10 = train_test_split(X10, y10, test_size = 0.2 , random_state = 50)
from sklearn.linear_model import LinearRegression
lin_reg10 = LinearRegression()
lin_reg10.fit(X_train10, y_train10)
y_pred10 = lin_reg10.predict(X_test10)

pop = [[338,112,5,5,5,9.5,1]]  
y_pred101 = lin_reg10.predict(pop)
print(y_pred101)

pickle.dump(lin_reg10, open('model1.pkl','wb'))
