import numpy as np
import time

def grid_search(diff_i: np.ndarray, i: int, perplexity: int) -> float:

    result = np.inf  # Set first result to be infinity

    norm = np.linalg.norm(diff_i, axis=1)
    std_norm = np.std(norm)  # Use standard deviation of norms to define search space

    for σ_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
        # Equation 1 Numerator
        p = np.exp(-(norm**2) / (2 * σ_search**2))

        # Set p = 0 when i = j
        p[i] = 0

        # Equation 1 (ε -> 0)
        ε = np.nextafter(0, 1)
        p_new = np.maximum(p / np.sum(p), ε)

        # Shannon Entropy
        H = -np.sum(p_new * np.log2(p_new))

        # Get log(perplexity equation) as close to equality
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            σ = σ_search

    return σ

def condi_prob(xdata, perplexity):
    n = len(xdata)
    c_pij = np.zeros((n, n))
    for i in range(n):
        dist = xdata[i] - xdata
        norm = np.linalg.norm(dist, axis=1)
        sigma_i = grid_search(dist, i, perplexity)
        c_pij[i, :] = np.exp(-(norm**2) / (2 * sigma_i**2))

        np.fill_diagonal(c_pij, 0)

        c_pij[i, :] = c_pij[i, :] / np.sum(c_pij[i, :])

    epsilon = np.nextafter(0, 1)
    c_pij = np.maximum(c_pij, epsilon)

    return c_pij

def p_prob(xdata, perplexity):
    n=len(xdata)
    c_pij = condi_prob(xdata, perplexity)
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

    qij = qij / qij.sum()

    epsilon = np.nextafter(0,1)
    qij = np.maximum(qij, epsilon)

    return qij

def gradient(pij, qij, ydata):
    n = len(pij)

    grad = np.zeros((n, ydata.shape[1]))
    for i in range(n):
        dist = ydata[i] - ydata
        g1 = np.array([(pij[i, :] - qij[i, :])])
        g2 = np.array([(1 + np.linalg.norm(dist, axis=1)**2) ** (-1)])
        grad[i] = 4 * np.sum((g1 * g2).T * dist, axis=0)
    return grad

def tsne(xdata, perplexity, max_iter, step, d):
    early_exaggeration = 20
    n = len(xdata)

    pij = p_prob(xdata, perplexity)

    Y = np.zeros(shape=(max_iter, n, d))
    Y_minus1 = np.zeros(shape=(n, d))
    Y[0] = Y_minus1
    Y1 = set_y(xdata, d)
    Y[1] = np.array(Y1)
    i = 1
    cost = 10
    alpha = 0.5
    while(i<max_iter-1):
        if i >= 250:

            alpha = 0.8
            early_exaggeration = 1

        qij = q_prob(Y[i])

        grad = gradient(early_exaggeration*pij, qij, Y[i])

        Y[i+1] = Y[i] - step * grad + alpha * (Y[i] - Y[i - 1])

        if i % 20 == 0 or i == 1:
            cost = np.sum(pij * np.log(pij / qij))
            print(f"Iteration {i}: Value of Cost Function is {cost}")
        i += 1
    return Y[-1], Y





