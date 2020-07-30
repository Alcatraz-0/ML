""" Dividing the data set in different types of clusters of data to perform further clustering and better understand those clusters with specific reason """


import pandas as pd 
import numpy as np 

dataset = pd.read_csv(r"C:\Users\anand\Desktop\STUDE\Visual_code\personal\python\ML\Clustering\week 08 assignment\datasets_721951_1255613_Country-data.csv")


#Child mortality and healthcare 

x_1 = dataset.iloc[:,[1, 3]].values

#Exports and Imports 

x_2 = dataset.iloc[:,[2, 4]].values

#Income and Inflation 

x_3 = dataset.iloc[:,[5, 6]].values

#life expectancy and Total fetility 

x_4 = dataset.iloc[:,[7, 8]].values

#healthcare and life expectancy 

x_5 = dataset.iloc[:,[3, 7]].values

#Child mortality and Total fetility 

x_6 = dataset.iloc[:,[1, 8]].values








