# Support Vector Regression Model

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r"Car_Purchasing_Data.csv", encoding = "windows-1252")
X = dataset.iloc[:, [4,5,7]].values
y = dataset.iloc[:, -1].values


#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Regression Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X_train, y_train) 

# Predicting the result

y_pred = regressor.predict(X_test)
y_pred = sc_y.inverse_transform(y_pred)
y_pred = y_pred.reshape(-1,1)
y_test = sc_y.inverse_transform(y_test)



#plotting the values

actual = plt.scatter(range(100), y_test, color = 'red', alpha = 0.4)
estm = plt.scatter(range(100), y_pred, color = 'blue', alpha = 0.4)
plt.title('Test Vs. Prediction (General)')
plt.xlabel('Features')
plt.ylabel('Expenditure')
plt.legend((actual, estm), ("Actual Values", "Predicted Values"))
plt.show()

for i in range(3):
    actual = plt.scatter(X_test[:, i], y_test, color = 'red', alpha = 0.4)
    estm = plt.scatter(X_test[:, i], y_pred, color = 'blue', alpha = 0.4)
    plt.title("Train vs Prediction")
    plt.xlabel("Features {}".format(i)), plt.ylabel("Expediture")
    plt.legend((actual, estm), ("Actual Values", "Predicted Values"))
    plt.show()
    
pred = pd.DataFrame(y_pred.flatten())
test = pd.DataFrame(y_test.flatten())
result = pd.concat([pred, test], axis=1)

pd.DataFrame(result).to_csv(r"predicted_values")
#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")
