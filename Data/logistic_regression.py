import json
import math
import os
import sys
import numpy as np
import random as rand

data_path = "data.npy"

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def dsigmoid(t):
    return np.exp(-t)/(1 + np.exp(-t))

def reg(data, y, alpha=0.1, num_iters=250):
    m, n = data.shape
    theta = np.zeros(len(data[0]))
    for i in range(num_iters):
        predictions = sigmoid(np.dot(data, theta))
        grad = np.dot(data.T,  predictions - y)
        theta = theta - alpha*grad

    return theta

def predict(x, theta, c):
    p = sigmoid(np.dot(x, theta))
    if(p > c):
        return 1
    else:
        return 0

def predict_p(x, theta):
    p = sigmoid(np.dot(x, theta))
    return p

def fit(data, theta):
    res = np.zeros(len(data))
    for k in range(len(data)):
        res[k] = predict(data[k], theta, 0.5)
    return res.astype(int)


if __name__ == "__main__":
    data = np.load(data_path)
    original = np.load(data_path)

    y = np.zeros(150)
    train = data[0:150, :]
    for i in range(len(train)):
        if(data[i][7] > 1):
            y[i] = 1

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(random_state=0, fit_intercept=False).fit(train, y)
    p1 = clf.predict(data[150:len(data), :]).astype(int)
    theta = clf.coef_.reshape(-1, 1)
    print(p1)
    print(clf.coef_, "\n------------")
    np.save("theta.npy", theta)
    theta = reg(train, y, 0.2, 2500)
    p2 = fit(data[150:len(data), :], clf.coef_.reshape(-1, 1))
    print(p2)
    print(theta)


    error = 0
    for i in range(len(p1)):
        if(p1[i] != p2[i]):
            error += 1
    print(error)














