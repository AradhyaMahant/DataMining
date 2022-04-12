import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(r'D:\upes\2nd year\Sem 2\Intro to ML\lab\Linear-Regression-Data.csv')
df.head()

x = np.array(df[['x']])
y = np.array(df[['y']])

x_mean=np.mean(x)
y_mean=np.mean(y)

num =0 
den =0
for i in range(len(x)):
    num += (x[i]-x_mean) + (y[i]-y_mean)
    den += (x[i]-x_mean) ** 2
     
b1=num/den  #coefficeint
b0=y_mean-(b1*x_mean)  #intercept

max_x= np.max(x)+100
min_x= np.min(x)-100


x1= np.linspace(min_x,max_x,1000)
y_pred=b0+b1*x
plt.plot(x1,y_pred,c="green")
plt.scatter(x,y,c="red")
plt.show()