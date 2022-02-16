#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv(r'D:\upes\2nd year\Data mining and prediction by machines\lab\Salary_Data.csv')
dataset = pd.read_csv(r'D:\upes\2nd year\Data mining and prediction by machines\lab\Salary_Data.csv')


X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

print(X)
print(Y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=.2, random_state=0)

print(x_train,'\n')
print(x_test,'\n')
print(y_train,'\n')
print(y_test,'\n')


from sklearn.linear_model import LinearRegression
LinearRegression
regression = LinearRegression()
regression.fit(x_train,y_train)
y_pred=regression.predict(x_test)
print('\n',y_pred)

plt.scatter(x_train,y_train, color='green')
plt.plot(x_train,regression.predict(x_train), color='red')
plt.title('Salary Vs Experience(Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('salary')
plt.show()
