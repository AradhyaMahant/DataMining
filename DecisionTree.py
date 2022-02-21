import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import category_encoders as ce
from sklearn.metrics import accuracy_score

dataset = pd.read_csv(r"D:\upes\2nd year\Sem 2\Intro to ML\lab\AllElectronics.csv")
dataset.head()

X = dataset.drop('buys_computer', axis=1)
y = dataset['buys_computer']

encoder = ce.OneHotEncoder()
X = encoder.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))

print(accuracy_score(y_test, classifier.predict(X_test))*100)