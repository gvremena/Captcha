import json
import math
import os
import numpy as np
from scipy import stats
from scipy.stats import multivariate_normal
from sklearn import preprocessing
from sklearn.cluster import KMeans

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture

data_path = "Data/data.npy"

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

def uniform_test(error):
    k, p = stats.kstest(error, "uniform", args=(min(error), max(error)))
    if p < 0.05:
        return 0
    else:
        return 1

def load_data(fpath, delete=False):
    with open(fpath, 'r') as json_file:
        new_data = json.load(json_file)
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

    distances = [[]]*len(mouse_pauses)
    for k in range(len(mouse_pauses)):
        f = mouse_pauses[k]
        l = []
        current_time = f[0][2]
        for i in range(len(f)):
            #print(f[i])
            if current_time == f[i][2]:
                l += [f[i]]
            if current_time != f[i][2] or i == len(f) - 1:
                distances[k] = distances[k] + [(total_distance(l), current_time)]
                current_time = f[i][2]
                l = [f[i]]

    for k in range(len(distances)):
        print(distances[k])

    normal = 0
    uniform = 0
    for i in range(len(mouse)):
        if normal_test(line_distances[i]) == 1:
            normal += 1
        if uniform_test(line_distances[i]) == 1:
            uniform += 1

    l = np.array([])
    l = np.append(l, correct/total) #
    l = np.append(l, pauses) # number of pauses
    l = np.append(l, np.mean(dist)) # average distance from center
    l = np.append(l, time[0]/1000) # start time
    l = np.append(l, np.mean(np.diff(time))/1000) # avg. time between clicks
    l = np.append(l, normal/len(mouse)) # % normal
    l = np.append(l, uniform/len(mouse)) # % uniform

    """if not any(np.equal(data, l).all(1)):
        data = np.append(data, np.array([l]), axis=0)
        np.save(data_path, data)
    """
    if (delete == True):
        os.remove(fpath)

    return data

def em_cluster(x, mu, sigma, p, num_iter=100):
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

if __name__ == "__main__":
    fpath = os.path.abspath("D:/Download/result.txt")

    if not os.path.exists(data_path):
        np.save("data/data.npy", np.empty((0, 6)))

    if os.path.exists(fpath):
        data = load_data(fpath, False)

"""
    #data = np.load(data_path)
    m = 200
    n = 3
    data = np.random.rand(m,n)*8
    k = 2;
    model = GaussianMixture(n_components=k, init_params='kmeans')
    model.fit(data)
    #print("gaussian mixtures")
    #print(model.n_iter_)
    #print(model.covariances_)
    #print(np.linalg.det(model.covariances_))
    print(model.predict(data))

    mu = KMeans(n_clusters=2, random_state=0, max_iter=5).fit(data).cluster_centers_ #data[np.random.choice(data.shape[0], k, replace=False), :]
    p = np.ones(k)/k

    sigma = np.array([datasets.make_spd_matrix(len(data[0])), datasets.make_spd_matrix(len(data[0]))])
    mu, sigma, p = em_cluster(data, mu, sigma, p, 25)
    res = fit(data, mu, sigma, p)
    print(res)
    print(sum(res))
    #print(sigma)
    #print(np.linalg.det(sigma))

    #y = [predict(x[i], mu, sigma, p) for i in range(len(data))]
    #print(y)
"""

