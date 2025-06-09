import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import utils

# Génère un cercle en 3D (N points)
def generate_circle_3d(N=100, radius=5):
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    z = radius * np.sin(2*angles)  # Pour donner un peu de structure 3D
    return np.stack((x, y, z), axis=1)

# Visualisation des données originales en 3D
def plot_3d(data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b')
    ax.set_title("Données initiales (cercle 3D)")
    plt.show()

# Visualisation du résultat t-SNE en 2D
def plot_2d(data, title="Projection t-SNE"):
    plt.figure()
    plt.scatter(data[:, 0], data[:, 1], c='r')
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.axis('equal')
    plt.show()

N = 100
X = generate_circle_3d(N)
plot_3d(X)

    # Paramètres t-SNE (tu peux les ajuster)
sigma = 2
max_iter = 4000

step = 1
d = 2

    # Exécution du t-SNE de ton utils.py
Y_final, Y_all = utils.tsne(X, sigma, max_iter, step, d)
plot_2d(Y_final, "Projection t-SNE")


