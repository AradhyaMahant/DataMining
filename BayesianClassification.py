import pandas as pd
import numpy as np 


data = pd.read_csv(r"D:\upes\2nd year\Sem 2\Intro to ML\lab\Bayes.csv")


X = data.iloc[:,:-1].values
print(X)


Y = data.iloc[: ,-1].values
print(Y)


from sklearn import preprocessing 
le = preprocessing.LabelEncoder()
Y = le.fit_transform(Y)
print(Y)


for i in range(0 , 4):
    X[:,i] = le.fit_transform(X[:,i])
print(X)


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
X_train , X_test , Y_train , Y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 1)
print(X_train)


model = GaussianNB()
Y_pred = model.fit(X_train, Y_train).predict(X_test)
print("Number of mislabeled points out of a total ",X_test.shape[0], "points: ",(Y_test != Y_pred).sum())



print("The X testing dataset is :\n",X_test)
print("The Y testing dataset is :\n",Y_test)
print("The predicted Values are :\n",Y_pred)