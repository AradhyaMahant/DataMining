import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'D:\upes\2nd year\Sem 2\Intro to ML\lab\diabetes.csv')
dataset.shape
dataset.head()

X = dataset.drop(['Outcome'], axis=1)
y = dataset['Outcome']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.svm import SVC
svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

