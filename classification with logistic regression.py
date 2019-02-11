import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

def sigmoid(x, theta):
    return 1 / (1 + np.exp(-np.dot(x, theta)))

def cost_function(theta, x, y):
    m = x.shape[0]
    return -(1 / m) * np.sum(
        y * np.log(sigmoid(x, theta)) + (1 - y) * np.log(
            1 - sigmoid(x, theta)))

def gradient_descent(theta, x, y):
    m = x.shape[0]
    return (1 / m) * np.dot(x.T, sigmoid(x, theta) - y)

def fit(x, y, theta):
        opt_weights = opt.fmin_tnc(func=cost_function, x0=theta,
                            fprime=gradient_descent, args=(x, y.flatten()))
        return opt_weights[0]
    
dataset = pd.read_csv("data.txt")
x = np.array((dataset.iloc[:,:-1]))
x = np.append(np.ones((100, 1)), x, axis=1)
y = dataset.iloc[:, -1]
admitted = dataset.loc[y==1]
not_admitted = dataset.loc[y==0]
y = y[:, np.newaxis]
theta = np.zeros((3, 1))
learning_rate = 0.001
theta = fit(x, y, theta)

x_values = [np.min(x[:, 1] - 5), np.max(x[:, 2] + 5)]
y_values = -(theta[0] + np.dot(theta[1], x_values))/theta[2]


plt.scatter(admitted.iloc[:, 0], admitted.iloc[:, 1], s=10, label='Admitted')
plt.scatter(not_admitted.iloc[:, 0], not_admitted.iloc[:, 1], s=10, label='Not Admitted')
plt.plot(x_values, y_values, label='Decision Boundary')
plt.xlabel('Marks in 1st Exam')
plt.ylabel('Marks in 2nd Exam')
plt.legend()
plt.show()
