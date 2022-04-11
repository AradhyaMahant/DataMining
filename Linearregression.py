import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\upes\2nd year\Sem 2\Intro to ML\lab\Linear-Regression-Data.csv')
df.head()

mean_x = sum(df['x']) / float(len(df['x']))
mean_y= sum(df['y']) / float(len(df['y']))

print(mean_x)
print(mean_y)


def variance(values, mean):
    return sum([(val-mean)**2 for val in values])
def covariance(x, mean_x, y , mean_y):
    covariance = 0.0
    for r in range(len(x)):
        covariance = covariance + (x[r] - mean_y) * (y[r] - mean_y)
    return covariance

variance_x, variance_y = variance(df['x'], mean_x), variance(df['y'], mean_y)
variance_x , variance_y

print(variance_x)
print(variance_y)

covariance_x_y = covariance(df['x'],mean_x,df['y'],mean_y)
covariance_x_y

print(covariance_x_y)