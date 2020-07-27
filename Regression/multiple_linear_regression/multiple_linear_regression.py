
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
#data regression
trainset = pd.read_csv(r'Train.csv')
x_train = trainset.iloc[:, :-1].values
y_train = trainset.iloc[:, -1].values

testtest = pd.read_csv(r'Test.csv')
x_test = testtest.iloc[:,:].values

#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")


#trainig the model
import statsmodels.regression.linear_model as sm
x_train = np.append(arr=np.ones((1600,1)).astype(int), values= x_train , axis=1 )
x_opt = x_train[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(y_train,x_opt).fit()
print(regressor_ols.summary())


#predicting the results
obtain = np.append(arr=np.ones((400, 1)).astype(int), values = x_test, axis = 1)
y_pred_OLS = regressor_ols.predict(obtain)
import pandas as pd 
pd.DataFrame(y_pred_OLS).to_csv(r"C:\file_path\multiple linear regression\predict_value.csv")

#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")

#plotting Train vs Prediction(Using OLS)
ones = []
for i in range(400):
    ones.append(i)
plt.scatter(ones[:400], y_train[:400], color = "blue", alpha=0.3)
plt.scatter(ones[:400], y_pred_OLS[:400], color = "red", alpha=0.3)
plt.title("Train vs Prediction(Using OLS)")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()


