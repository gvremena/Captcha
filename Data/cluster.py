import json
import math
import os
import sys
import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

data_path = "data.npy"

def linedist(a, b, t):
    return abs((-a*t[0] + t[1] - b))/(math.sqrt(a*a + 1))

def sigmoid(t):
    return 1/(1 + np.exp(-t))

def dsigmoid(t):
    return np.exp(-t)/(1 + np.exp(-t))

def stochasticGradient(x, y, alpha=0.01, num_iters=200):
    m, n = x.shape
    theta = np.zeros(n)

    for k in range(num_iters):
        i = np.random.randint(m)
        theta = theta + (alpha/m)*(y[i]*x[i]*dsigmoid(theta.T.dot(x[i])) + (-1 + y[i])*x[i]*sigmoid(theta.T.dot(x[i])))

    return theta

def predict(x, mu, sigma, p):
    k = len(mu)
    pr = np.zeros(k)
    for i in range(k):
        pr[i] = p[i]*multivariate_normal.pdf(x, mu[i], sigma[i])
    return np.argmax(pr)

def fit(data, mu, sigma, p):
    res = np.zeros(len(data))
    for k in range(len(data)):
        res[k] = predict(data[k], mu, sigma, p)
    return res.astype(int)

def normal_test(error):
    k, p = stats.normaltest(error)
    if p < 0.05:
        return 0
    else:
        return 1

def flatten(l):
    return [item for sublist in l for item in sublist]

def total_distance(l):
    d = 0
    for i in range(1, len(l)):
        d += np.sqrt((l[i][0] - l[i - 1][0])*(l[i][0] - l[i - 1][0]) + (l[i][1] - l[i - 1][1])*(l[i][1] - l[i - 1][1]))
    return d

def distance(t1, t2):
    return np.sqrt((t1[0] - t2[0])*(t1[0] - t2[0]) + (t1[1] - t2[1])*(t1[1] - t2[1]))

def uniform_test(error):
    k, p = stats.kstest(error, "uniform", args=(min(error), max(error)))
    if p < 0.05:
        return 0
    else:
        return 1

def print_data():
    data = np.load(data_path)
    print(data)

def load_data(input_json):
    new_data = input_json
    data = np.load(data_path)

    correct = new_data["correct"]
    total = new_data["total"]
    dist = new_data["dist"]
    clicks = new_data["clicks"]
    time = new_data["timer"]
    mouse = new_data["mouse"]
    mouse_flat = [i for sublist in mouse for i in sublist]

    line_distances =[[0 for j in range(len(mouse[i]))] for i in range(len(mouse))]
    pauses = 0

    for i in range(len(mouse)):
        start = mouse[i][0]
        end = mouse[i][len(mouse[i]) - 1]
        a = (end[1] - start[1])/(end[0] - start[0])
        b = start[1] - a*start[0]
        for j in range(len(mouse[i])):
            line_distances[i][j] = linedist(a, b, mouse[i][j])

    mouse_pauses = [[]]
    for i in range(len(mouse_flat)):
        diff = mouse_flat[i][2] - mouse_flat[i - 1][2]
        if diff >= 300:
            pauses += 1
            mouse_pauses += [[mouse_flat[i]]]
        else:
            mouse_pauses[len(mouse_pauses) - 1] = mouse_pauses[len(mouse_pauses) - 1] + [mouse_flat[i]]

    velocity = [[]]*len(mouse_pauses)
    acceleration = [[]]*len(mouse_pauses)
    for k in range(len(mouse_pauses)):
        f = mouse_pauses[k]
        for i in range(1, len(f)):
            if(len(f)) > 1:
                dtime = (f[i][2] - f[i - 1][2])/1000
                ddist = distance(f[i], f[i - 1])
                velocity[k] = velocity[k] + [ddist/dtime]
                if i > 1:
                    acceleration[k] = acceleration[k] + [np.abs(ddist/dtime - velocity[k][i - 2])/dtime]
            elif len(f) <= 2:
                acceleration[k] = [0]
            elif len(f) <= 1:
                velocity[k] = [0]

    print(np.mean(flatten(acceleration)))
    """normal = 0
    uniform = 0
    for i in range(len(mouse)):
        if normal_test(line_distances[i]) == 1:
            normal += 1
        if uniform_test(line_distances[i]) == 1:
            uniform += 1
    """

    l = np.array([])
    l = np.append(l, correct/total) #
    l = np.append(l, pauses) # number of pauses
    l = np.append(l, np.var(dist)) # distance between center
    l = np.append(l, time[0]/1000) # start time
    l = np.append(l, time[len(time) - 1]/1000) # end time
    l = np.append(l, np.var(np.diff(time))/1000) # time between click
    l = np.append(l, np.var(flatten(velocity)))
    l = np.append(l, np.var(flatten(line_distances)))
    l = np.append(l, np.abs(np.var(flatten(acceleration))))
    #l = np.append(l, normal/len(mouse)) # % normal
    #l = np.append(l, uniform/len(mouse)) # % uniform

    if len(data) == 0 or (not any(np.equal(data, l).all(1))):
        if len(data) == 0:
            data = np.array([l])
        else:
            data = np.append(data, np.array([l]), axis=0)

        np.save(data_path, data)

    #if (delete == True):
    #    os.remove(fpath)

    return l

def em_cluster(x, mu, sigma, p, num_iter=10):
    k = len(p)
    m = len(x)
    n = len(x[0])
    gamma = np.zeros((m, k))

    for e in range(num_iter):
        # E step
        for i in range(m):
            b = 0
            for l in range(k):
                sigma_inv = np.linalg.inv(sigma[l])
                prod = np.dot(x[i] - mu[l], np.dot(sigma_inv, (x[i] - mu[l]).reshape(-1, 1)))
                b += p[l]/(np.power(2*np.pi, n/2)*np.sqrt(np.linalg.det(sigma[l])))*np.exp(-prod)
                #b += p[l]*multivariate_normal.pdf(x[i], mu[l], sigma[l])
            for j in range(k):
                sigma_inv = np.linalg.inv(sigma[j])
                det = np.linalg.det(sigma[j])
                prod = np.dot(x[i] - mu[j], np.dot(sigma_inv, (x[i] - mu[j]).reshape(-1, 1)))
                a = p[j]/(np.power(2*np.pi, n/2)*np.sqrt(det))*np.exp(-prod)
                #a = p[j]*multivariate_normal.pdf(x[i], mu[j], sigma[j])
                gamma[i][j] = a/b

        # M step
        s = np.sum(gamma, axis=0)
        mu_new = np.zeros(np.shape(mu))
        sigma_new = np.zeros(np.shape(sigma))
        for j in range(k):
            for i in range(m):
                mu_new[j] += gamma[i][j]*x[i]/s[j]
                sigma_new[j] += gamma[i][j]*np.dot((x[i] - mu[j]).reshape(-1, 1), (x[i] - mu[j]).reshape(1, -1))/s[j]
        p = s/m
        sigma = sigma_new
        mu = mu_new
    return mu, sigma, p

def kmeans():
    pass

if __name__ == "__main__":
    fpath = os.path.abspath("D:/Download/result.txt")

    if not os.path.exists(data_path):
        np.save("data.npy", np.empty((0, 6)))

    data = np.load(data_path)
    original = np.load(data_path)

    print(np.shape(data))
    k = 2
    mu = KMeans(n_clusters=2, random_state=0, max_iter=5).fit(data).cluster_centers_ #data[np.random.choice(data.shape[0], k, replace=False), :]
    p = np.ones(k)/k
    sigma = np.array([datasets.make_spd_matrix(len(data[0])), datasets.make_spd_matrix(len(data[0]))])
    #sigma = np.array([np.diag(np.var(data, axis=0)), np.diag(np.var(data, axis=0))])
    #sigma = np.array([np.cov(data[random.sample(range(len(data)), 9), :]), np.cov(data[random.sample(range(len(data)), 9), :])])
    print("Pocinje")
    #mu, sigma, p = em_cluster(data, mu, sigma, p, 20)


    model = GaussianMixture(n_components=2)
    model.fit(data)
    sigma = model.covariances_
    p = model.weights_
    mu = model.means_

    print(sigma)
    print(mu)
    print(p)
    np.save("mu.npy", mu)
    np.save("sigma.npy", mu)
    np.save("p.npy", mu)

    print(model.predict(data))
    #print(fit(data, mu, sigma, p))

    print(KMeans(n_clusters=2, random_state=0).fit_predict(data))

    from sklearn.linear_model import LogisticRegression
    y = np.zeros(150)
    train = data[0:150, :]
    for i in range(len(train)):
        if(data[i][7] > 1):
            y[i] = 1

    clf = LogisticRegression(random_state=0).fit(train, y)
    print(clf.predict(data[150:len(data), :]))


    #res = fit(data, mu, sigma, p)
    #print(res)
    #print(sum(res))
    #print(sigma)
    #print(np.linalg.det(sigma))

    #y = [predict(x[i], mu, sigma, p) for i in range(len(data))]
    #print(y)"""

