import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r"D:\upes\2nd year\Sem 2\Intro to ML\lab\insurance_data.csv")


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

ml = LogisticRegression()

ml.fit(x_train,y_train)

y_pred = ml.predict(x_test)

plt.scatter(x_test,y_test,color= 'red', marker='+')
plt.scatter(x_test,y_pred,color='blue', marker='.')
plt.xlabel("Age of person")
plt.ylabel("Bought Insurance 1=Bought 0=Did not Buy")

print(ml.score(x_test,y_test))

print(confusion_matrix(y_test,y_pred))

