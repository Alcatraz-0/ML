
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt




x_train = pd.read_csv(r'C:\\Visual_code\personal\python\ML\Classification\K-Nearest Neighbor\Diabetes_XTrain.csv')
y_train = pd.read_csv(r'C:\\Visual_code\personal\python\ML\Classification\K-Nearest Neighbor\Diabetes_YTrain.csv')
x_test  = pd.read_csv(r'C:\\Visual_code\personal\python\ML\Classification\K-Nearest Neighbor\Diabetes_Xtest.csv')


#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")


from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

from sklearn.neighbors import KNeighborsClassifier
knc = KNeighborsClassifier(n_neighbors=5, metric = 'minkowski', p = 2)
knc.fit(x_train, y_train.values.ravel())


y_pred = knc.predict(x_test)


pred = pd.DataFrame(y_pred.flatten())
pd.DataFrame(pred).to_csv(r"C:\\Visual_code\personal\python\ML\Classification\K-Nearest Neighbor\predicted_values_2.csv")
#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")







