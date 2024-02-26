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

## Output:
df.head()
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/8475ec01-1fb9-4cc3-a128-496c88362dfe)

df.tail()
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/4d894140-81d1-4077-bd02-8d70af4a8ad0)

Array values of X
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/863280ad-fec5-4120-bb07-2c63347e3fa4)

Array values of Y
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/c41fb50b-e4d9-413e-b3fd-79feff48dd0c)

Values of Y prediction
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/21ffa343-1b0f-4a83-8a14-1f4c0372ce35)

Array values of Y test
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/dc24f8b5-fefe-499d-8fc6-3bed892343cd)

Training Set Data
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/b3776d9f-f2b3-4fed-a718-8d87809b9504)

Test Set Data
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/577e6f2c-42d3-4941-a394-f1b81293bee2)

Values of MSE, MAE and RMSE
![image](https://github.com/23004426/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/144979327/e97fcfd6-0185-48e0-8f51-5a95154d334a)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
