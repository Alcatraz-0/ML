""" Dividing the data set in different types of clusters of data to perform further clustering and better understand those clusters with specific reason """


import pandas as pd 
import numpy as np 

dataset = pd.read_csv(r"datasets_721951_1255613_Country-data.csv")

'''sometimes using file names works but when it doesn't one can use this method too
x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")'''


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








