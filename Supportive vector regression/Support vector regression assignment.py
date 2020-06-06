# Support Vector Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"Car_Purchasing_Data.csv", encoding = "windows-1252")
X = dataset.iloc[:, [2,3,4,5,6,7]].values
y = dataset.iloc[:, -1].values

#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
x = sc_X.fit_transform(X.reshape(-1,1))
y = sc_y.fit_transform(y.reshape(-1,1))

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train.ravel())

# Predicting the result

y_pred = sc_y.inverse_transform(regressor.predict(X_test))

# Plotting the results vs test
ones = []
for i in range(100):
    ones.append(i)
plt.plot(ones[:100], sc_y.inverse_transform(y_train[:100]), color = "blue", alpha=0.3)
plt.plot(ones[:100], y_pred[:100], color = "red", alpha=0.3)
plt.title("prediction vs test")
plt.xlabel("People in test set"), plt.ylabel("Purchase power")
plt.show()


