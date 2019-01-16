import numpy as np
import matplotlib.pyplot as plt

def differentials(x, y, theta):
    return np.dot((np.dot(x, theta) - y), x)/4
    
def find_error(x, y, theta):
    return np.sum((np.dot(x, theta) - y)**2)


x = np.array([[1.0, 2104.0, 5.0, 1.0, 45.0],
              [1.0, 1416.0, 3.0, 2.0, 40.0],
              [1.0, 1534.0, 3.0, 2.0, 30.0],
              [1.0, 852.0, 2.0, 1.0, 36.0]])
x = 1.0 / x

y = np.array([460.0, 232.0, 315.0, 178.0])
theta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
learning_rate = 0.001

#Gradient descent
next_e_total = find_error(x, y, theta)

while True:
    e_total = next_e_total
    theta -= learning_rate * differentials(x, y, theta)    
    next_e_total = find_error(x, y, theta)
    print("Theta[0]=" + str(round(theta[0], 3)) + " Theta[1]=" + str(round(theta[1], 3)) +
      " Theta[2]=" + str(round(theta[2], 3)) + " Theta[3]=" + str(round(theta[3], 3)) + " Theta[4]= " + str(round(theta[4], 3)))
    print("Error = " + str(e_total))
    
    if e_total < next_e_total:
        break
        
print("Theta[0]=" + str(round(theta[0], 3)) + " Theta[1]=" + str(round(theta[1], 3)) +
      " Theta[2]=" + str(round(theta[2], 3)) + " Theta[3]=" + str(round(theta[3], 3)) + " Theta[4]= " + str(round(theta[4], 3)))
print("Error = " + str(e_total))

