import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def differential(x, y, theta):
    return np.dot((np.dot(x, theta) - y), x)/100
    
def find_error(x, y, theta):
    return np.sum((np.dot(x, theta) - y)**2)


#Training dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
df = df.sample(frac=1).reset_index(drop=True)
y = df.loc[0:99, 4]
for i in range(0, y.size):
    if y[i] == 'Iris-setosa':
        y[i] = 1
    else:
        y[i] = 0

x_training = df.loc[0:79, 0:3]
x_training = np.hstack((np.ones((80, 1)), x_training))
y_training = df.loc[0:79, 4]
x_testing = df.loc[80:99, 0:3]
x_testing = np.hstack((np.ones((20, 1)), x_testing))
y_testing = df.loc[80:99, 4]

theta = np.ones(5)
learning_rate = 0.01

#Training
next_e_total = find_error(x_training, y_training, theta)
while True:
    e_total = next_e_total
    theta = theta - learning_rate * differential(x_training, y_training, theta)    
    next_e_total = find_error(x_training, y_training, theta)
    print("Theta[0]=" + str(round(theta[0], 3)) + " Theta[1]=" + str(round(theta[1], 3)) +
      " Theta[2]=" + str(round(theta[2], 3)) + " Theta[3]=" + str(round(theta[3], 3)) + " Theta[4]= " + str(round(theta[4], 3)))
    print("Error = " + str(round(e_total, 5)))
    
    if round(e_total, 4) <= round(next_e_total, 4):
          break

