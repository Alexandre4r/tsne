import numpy as np
import time

def condi_prob(xdata, sigma):
    n = len(xdata)
    c_pij = np.zeros((n, n))
    for i in range(n):
        dist = xdata[i] - xdata
        norm = np.linalg.norm(dist, axis=1)
        c_pij[i, :] = np.exp(-(norm**2) / (2 * sigma**2))

        np.fill_diagonal(c_pij, 0)

        c_pij[i, :] = c_pij[i, :] / np.sum(c_pij[i, :])

    epsilon = np.nextafter(0, 1)
    c_pij = np.maximum(c_pij, epsilon)

    return c_pij

def p_prob(xdata, sigma):
    n=len(xdata)
    c_pij = condi_prob(xdata, sigma)
    pij = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1, n):
            pij[i,j] = (c_pij[i,j] + c_pij[j,i]) / (2 * n)
            pij[j, i] = pij[i,j]
    epsilon = np.nextafter(0, 1)
    pij = np.maximum(pij, epsilon)
    return pij

def set_y(xdata, d = 2):
    return np.random.normal(size=(len(xdata), d))

def q_prob(ydata):
    n=len(ydata)
    qij = np.zeros((n,n))
    for i in range(n):
        dist = ydata[i] - ydata
        norm = np.linalg.norm(dist, axis = 1)
        qij[i,:] = (1 + norm**2) ** (-1)

    np.fill_diagonal(qij, 0)

    epsilon = np.nextafter(0,1)
    qij = np.maximum(qij, epsilon)

    return qij

def gradient(pij, qij, ydata):
    n = len(pij)

    grad = np.zeros((n, ydata.shape[1]))
    for i in range(n):
        dist = ydata[i] - ydata
        g1 = np.array([(pij[i, :] - qij[i, :])])
        g2 = np.array([(1 + np.linalg.norm(dist, axis=1)) ** (-1)])
        grad[i] = 4 * np.sum((g1 * g2).T * dist, axis=0)
    return grad

def tsne(xdata, sigma, max_iter, step, d):
    n = len(xdata)

    pij = p_prob(xdata, sigma)

    Y = np.zeros(shape=(max_iter, n, d))
    Y_minus1 = np.zeros(shape=(n, d))
    Y[0] = Y_minus1
    Y1 = set_y(xdata, d)
    Y[1] = np.array(Y1)

    for i in range(1, max_iter - 1):
        qij = q_prob(Y[i])

        grad = gradient(pij, qij, Y[i])

        Y[i+1] = Y[i] - step * grad + 0.8 * (Y[i] - Y[i - 1])

        if i % 50 == 0 or i == 1:
            cost = np.sum(pij * np.log(pij / qij))
            print(f"Iteration {i}: Value of Cost Function is {cost}")

    return Y[-1], Y


