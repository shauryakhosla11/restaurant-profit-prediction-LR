# linear_regression.py
import numpy as np
import matplotlib.pyplot as plt
import copy
import math
from utils import load_data  # Make sure utils.py is in the same folder

# ========== Load and Visualize Data ==========

x_train, y_train = load_data()

print("Type of x_train:", type(x_train))
print("First five elements of x_train:\n", x_train[:5])
print("First five elements of y_train:\n", y_train[:5])
print("The shape of x_train is:", x_train.shape)
print("The shape of y_train is:", y_train.shape)
print("Number of training examples (m):", len(x_train))

# Plotting data
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# ========== Compute Cost Function ==========

def compute_cost(x, y, w, b): 
    m = x.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = w * x[i] + b
        cost += (f_wb - y[i]) ** 2
    return cost / (2 * m)

# ========== Compute Gradient ==========

def compute_gradient(x, y, w, b): 
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_db += f_wb - y[i]
        dj_dw += (f_wb - y[i]) * x[i]
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db

# ========== Gradient Descent ==========

def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    m = len(x)
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)
    b = b_in
    
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        if i < 100000:
            cost = cost_function(x, y, w, b)
            J_history.append(cost)

        if i % math.ceil(num_iters / 10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}")

    return w, b, J_history, w_history

# ========== Train Model ==========

initial_w = 0.
initial_b = 0.
iterations = 1500
alpha = 0.01

w, b, _, _ = gradient_descent(x_train, y_train, initial_w, initial_b,
                              compute_cost, compute_gradient, alpha, iterations)

print("Parameters found by gradient descent: w = {:.4f}, b = {:.4f}".format(w, b))

# ========== Make Predictions ==========

m = x_train.shape[0]
predicted = np.zeros(m)

for i in range(m):
    predicted[i] = w * x_train[i] + b

# Plot fit
plt.plot(x_train, predicted, c="b")
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Profits vs. Population per city")
plt.ylabel('Profit in $10,000')
plt.xlabel('Population of City in 10,000s')
plt.show()

# Custom predictions
predict1 = 3.5 * w + b
predict2 = 7.0 * w + b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1 * 10000))
print('For population = 70,000, we predict a profit of $%.2f' % (predict2 * 10000))
