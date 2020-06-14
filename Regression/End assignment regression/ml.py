#importing libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler



#importing the data set 

dataset = pd.read_csv(r"C:\\python\ML\regression\End assignment regression\datasets_88705_204267_Real estate.csv")
x = dataset.iloc[:,1:-1]
y = dataset.iloc[:,-1]
#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")

#plotting the findings 
#relation of price with metro station 
mtr_dst_st = x.sort_values("X3 distance to the nearest MRT station", axis = 0, ascending = True, 
                 inplace = False, na_position ='last')
hs_prc_st = y.sort_values(inplace = False, na_position ='last')
mtr_dst_st = mtr_dst_st.to_numpy()
plt.plot(mtr_dst_st[:,2:3], hs_prc_st, color = "green")
plt.title("Relation of price with metro station")
plt.xlabel("Distance from metro station")
plt.ylabel("House price")
plt.show()

#relation of price with age 
mtr_dst_st = x.sort_values("X2 house age", axis = 0, ascending = True, 
                 inplace = False, na_position ='last')
hs_prc_st = y.sort_values(inplace = False, na_position ='last')
mtr_dst_st = mtr_dst_st.to_numpy()
plt.plot(mtr_dst_st[:,1:2], hs_prc_st, color = "green")
plt.title("relation of price with age ")
plt.xlabel("Age of the house")
plt.ylabel("House price")
plt.show()

#relation of price with no of convenience store 
mtr_dst_st = x.sort_values("X4 number of convenience stores", axis = 0, ascending = True, 
                 inplace = False, na_position ='last')
hs_prc_st = y.sort_values(inplace = False, na_position ='last')
mtr_dst_st = mtr_dst_st.to_numpy()
plt.plot(mtr_dst_st[:,3:4], hs_prc_st, color = "green")
plt.title("relation of price with no of convenience store")
plt.xlabel("No of convenience store")
plt.ylabel("House price")
plt.show()



#Scaling the dataset 
sc_x = StandardScaler()
sc_y = StandardScaler()
x_t = sc_x.fit_transform(x)
y_t = sc_y.fit_transform(y.values.reshape(-1, 1))


#splitting the data scaled set
from sklearn.model_selection import train_test_split
xt_train, xt_test, yt_train, yt_test = train_test_split(x_t, y_t, test_size = 0.2, random_state = 0)


#splitting the data set 
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)


#training the linear regression model
from sklearn.linear_model import LinearRegression
regressor_lin = LinearRegression()
regressor_lin.fit(xt_train, yt_train)

#predicting the results and accuracies of linear regression model 
lin_pred = regressor_lin.predict(xt_test) 
from sklearn.metrics import mean_squared_error as mse
from math import sqrt
error_linear = mse(yt_test,lin_pred)
rmse_linear  = sqrt(error_linear)
r_sq_linear  = regressor_lin.score(xt_train,yt_train)
print(f"The rmse of linear regression: {rmse_linear}")
print(f"The error of linear regression: {error_linear}")
print(f"The r square of linear regression: {r_sq_linear}")



#training the multiple linear regression model
import statsmodels.regression.linear_model as sm
#Backwards elimnination
x_opt = xt_train[:,[0,1,2,3,4,5]]
regressor_ols = sm.OLS(yt_train,x_opt).fit()
print(regressor_ols.summary())

x_opt = xt_train[:,[0,1,2,3,4]]
regressor_ols = sm.OLS(yt_train,x_opt).fit()
print(regressor_ols.summary())

#predicting the new results 
y_pred_OLS = regressor_ols.predict(xt_test[:,[0,1,2,3,4]])


#predicting the results and accuracies of Multiple linear regression model 
error_mult = mse(y_test,y_pred_OLS)
rmse_mult  = sqrt(error_linear)
r_sq_mult  = regressor_lin.score(x_train,y_train)
print(f"The rmse of Multiple linear regression: {rmse_linear}")
print(f"The error of Multiple linear regression: {error_linear}")
print(f"The r square of Multiple linear regression: {r_sq_linear}")


#implementing polynomial regression 
from sklearn.preprocessing import PolynomialFeatures
poly_regresssor = PolynomialFeatures(degree=10)
x_poly = poly_regresssor.fit_transform(x_train)
poly_lin_regressor = LinearRegression()
poly_lin_regressor.fit(x_poly, y_train)

#predicting polynomial regression
plr = poly_lin_regressor.predict(poly_regresssor.fit_transform([[2013.583,17.4,6488.021,1,24.95719,121.47353]]))
y_pred_poly = poly_lin_regressor.predict(poly_regresssor.fit_transform(x_test))

print(f"The the predicted value using polynomial regression  {plr} ")
#[[2013.1583,17.4,6488.021,1,24.95719,121.47353]]


# Fitting the support vector regression model 
from sklearn.svm import SVR
regressor_svm = SVR(kernel='rbf')
regressor_svm.fit(xt_train, yt_train) 

#predicting support vector regression model
y_pred_svm = regressor_svm.predict(x_test)
y_pred_svm = y_pred_svm.reshape(-1,1)



#fitting decision tree regression 
from sklearn.tree import DecisionTreeRegressor
regressor_dtr = DecisionTreeRegressor()
regressor_dtr.fit(x_train,y_train)

#predicting using  decision tree regression 
dtr = regressor_dtr.predict([[2013.583,17.4,6488.021,1,24.95719,121.47353]])
print(f"The the predicted value using decision tree regression   {dtr} ")
y_pred_dtr = regressor_dtr.predict(x_test)

#fitting Random Forest regression 
from sklearn.ensemble import RandomForestRegressor
regressor_rfr = RandomForestRegressor(n_estimators=10 ,random_state = 0)
regressor_rfr.fit(x_train, y_train)

#predicting using Random Forest regression 
rfr = regressor_rfr.predict([[2013.583,17.4,6488.021,1,24.95719,121.47353]])
print(f"The the predicted value using Random Forest Regression   {rfr} ")
y_pred_rfr = regressor_rfr.predict(x_test)



#Applying inverse and type conversions to expost in csv file
lin_pred = sc_y.inverse_transform(lin_pred)
y_test = y_test.to_numpy()
y_pred_OLS = sc_y.inverse_transform(y_pred_OLS)
y_pred_svm = sc_y.inverse_transform(y_pred_svm)

#converting to 1 d 
lin_pred = pd.DataFrame(lin_pred.flatten())
mult_pred = pd.DataFrame(y_pred_OLS.flatten())
svm_pred = pd.DataFrame(y_pred_OLS.flatten())
poly_pred = pd.DataFrame(y_pred_poly.flatten())
dtr_pred  = pd.DataFrame(y_pred_dtr.flatten())
rfr_pred = pd.DataFrame(y_pred_rfr.flatten())
test = pd.DataFrame(y_test.flatten())


#concatnating the results
result = pd.concat([lin_pred, mult_pred, svm_pred, poly_pred, dtr_pred, rfr_pred, test], axis=1)

#exporting to csv
pd.DataFrame(result).to_csv(r"C:\\python\ML\regression\End assignment regression\predicted_values.csv")
#sometimes using file names works but when it doesn't one can use this method too
#x_test  = pd.read_csv(r"C:\file_path\python\ML\regression\Linear_X_Test.csv")


#Train vs Prediction(Using linear regression)
ones = []
for i in range(80):
    ones.append(i)
est = plt.scatter(ones[:80], lin_pred[:80], color = "red", alpha=0.3)
real = plt.scatter(ones[:80], test[:80], color = "blue")
plt.title("Train vs Prediction(Using linear regression)")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()


#Train vs Prediction(Using multiple linear regressions)
plt.scatter(ones[:80], mult_pred[:80], color = "green", alpha=0.3)
plt.scatter(ones[:80], test[:80], color = "blue")
plt.title("Train vs Prediction(Using multiple linear regressions)")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()

#Train vs Prediction(Using Standard vector (svm))
plt.scatter(ones[:80], svm_pred[:80], color = "black", alpha=0.3)
plt.scatter(ones[:80], test[:80], color = "blue")
plt.title("Train vs Prediction(Using Standard vector (svm))")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()

#Train vs Prediction(Using polynomial regression)
plt.scatter(ones[:80], poly_pred[:80], color = "cyan", alpha=0.3)
plt.scatter(ones[:80], test[:80], color = "blue")
plt.title("Train vs Prediction(Using polynomial regression)")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()

#Train vs Prediction(Using random forest regression)
plt.scatter(ones[:80], rfr_pred[:80], color = "orange", alpha=0.3)
plt.scatter(ones[:80], test[:80], color = "blue")
plt.title("Train vs Prediction(Using random forest regression)")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()

#Train vs Prediction(Using decision tree regression
plt.scatter(ones[:80], dtr_pred[:80], color = "brown", alpha=0.3)
plt.scatter(ones[:80], test[:80], color = "blue")
plt.title("Train vs Prediction(Using decision tree regression)")
plt.xlabel("Features"), plt.ylabel("Target")
plt.show()

