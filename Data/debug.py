import numpy as np
import logistic_regression as lg

data_path = "data.npy"

if __name__ == "__main__":
    theta = np.load("theta.npy");
    print(theta)

    data = np.load(data_path)

    p2 = lg.fit(data[150:len(data), :], clf.coef_.reshape(-1, 1))
    print(p2)