import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r'D:\upes\2nd year\Sem 2\Intro to ML\lab\age_salary.csv')
dataset.head(15)

x= dataset.drop(['Purchase_Item', 'Index'], axis=1)
y = dataset['Purchase_Item']

print(x.isnull().sum())
print(" ")


x['Age'].fillna(x['Age'].mean(),inplace=True)
x['Salary'].fillna(x['Salary'].mean(), inplace=True)
print(x.isnull().sum())

encoder = ce.OneHotEncoder(cols=['Nation'])

x = encoder.fit_transform(x)


x_predict = x.iloc[13]
print(x_predict)
y_predict = y.iloc[13]
print(y_predict)

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,random_state = 42)
x_train = np.asarray(x_train)
x_test = np.asarray(x_test)
y_train = np.asarray(y_train)
y_test = np.asarray(y_test)
x_predict = np.asarray(x_predict).reshape(1,-1)

knn= KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train,y_train)
print("The accuracy of the classifier is: {}%".format(accuracy_score(y_test, knn.predict(x_test))*100))

if(knn.predict(x_predict)==1):
    print("When {0}, Purchase_Item = {1}".format("National=India, Age=45 and Salary=50000",'Yes'))
else :
    print("When {0}, Purchase_Item = {1}".format("National=India, Age=25 and Salary=36000",'Yes'))