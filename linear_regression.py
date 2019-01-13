import numpy as np
import matplotlib.pyplot as plt


def differential1(theta, x, y):
        sum = 0.0
        for i in range(0, x.size):
                sum += 2 * (theta[0] + theta[1]*x[i] - y[i])
        return sum / (2 * x.size)

def differential2(theta, x, y):
        sum = 0.0
        for i in range(0, x.size):
                sum += 2 * x[i] * (theta[0] + theta[1]*x[i] - y[i])
        return sum / (2 * x.size)

def find_error(x, y, theta):
        e = 0
        for i in range(0, x.size):
                e += (theta[0] + theta[1]*x[i] - y[i])**2
        return e


x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([6.0, 5.0, 7.0, 10.0])
theta = np.array([1.0, 1.0])
learning_rate = 0.01

#Gradient descent
e_total = find_error(x, y, theta)
while True:   
        prev_e = e_total
        theta[0] -= learning_rate * differential1(theta, x, y)
        theta[1] -= learning_rate * differential2(theta, x, y)
        e_total = find_error(x, y, theta)
        if e_total > prev_e:
                break
        
print("Theta[0] = " + str(round(theta[0], 1)) + " Theta[1] = " + str(round(theta[1], 1)))
print("Total error = " + str(round(e_total, 1)))


plt.plot(x, y, "ro")
plt.plot(x, theta[0] + theta[1]*x)
plt.show()
