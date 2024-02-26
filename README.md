# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values and import linear regression from sklearn.
3. Assign the points for representing in the graph.
4. Predict the regression for marks by using the representation of the graph and compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Tirupathi Jayadeep
RegisterNumber:  212223240169

import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
x=(df.iloc[:,:-1]).values
print(x)
y=(df.iloc[:,1]).values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color="orange")
plt.plot(x_train,regressor.predict(x_train),color="blue")
plt.title("Hours Vs Scores(Train Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color="purple")
plt.plot(x_test,regressor.predict(x_test),color="yellow")
plt.title("Hours vs scores (test set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```

### Output:
df.head()

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/d7cb026f-f5cd-4c69-90b8-770a826f3630)


df.tail()

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/42b1ba7c-f813-4531-bf95-20c6bb9f04e1)


Array values of X

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/5d59a8bd-d2bd-4eb8-adfe-ce1e310451af)


Array values of Y

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/63f01cde-4665-4e6f-ad42-092b2f288478)


Values of Y prediction

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/583bf09b-f2c6-4cec-9e21-19e51cb8c0db)


Array values of Y test

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/17c17edc-8be7-405e-8326-9f498e4d2058)


Training Set Data

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/86415740-0504-418e-9e89-ede39cb56fca)


Test Set Data

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/aa100f7c-c55a-40dd-9e44-599d3ce1351c)


Values of MSE, MAE and RMSE

![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/d2833dcd-42ab-4d5a-a625-caa7f692b9b7)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
