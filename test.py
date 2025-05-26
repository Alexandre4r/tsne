import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from utils import *
# 1. Charger et réduire les données
print("Chargement des données...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X_total = pd.DataFrame(mnist["data"])
y_total = pd.DataFrame(mnist["target"]).astype(int)

# 2. Réduction : choisir 1000 points et réduire à 30 dimensions
sample_indices = np.random.choice(len(X_total), size=1000, replace=False)
X_reduced = X_total.iloc[sample_indices]
y_reduced = y_total.iloc[sample_indices].values.flatten()

print("Réduction PCA...")
X_pca = PCA(n_components=30).fit_transform(X_reduced)

# 3. Appliquer ton t-SNE
print("Lancement de t-SNE personnalisé...")
Y_final, all_Y = tsne(
    xdata=X_pca,
    sigma=5,           # à ajuster si besoin
    max_iter=300,      # nombre d’itérations
    step=0.5,          # pas de descente
    d=2                # on réduit à 2 dimensions
)

# 4. Affichage en 2D
print("Affichage du résultat...")
plt.figure(figsize=(10, 8))
scatter = plt.scatter(Y_final[:, 0], Y_final[:, 1], c=y_reduced, cmap='tab10', s=10, alpha=0.7)
plt.colorbar(scatter, ticks=range(10), label="Classe (chiffre)")
plt.title("t-SNE personnalisé sur MNIST (1000 points)")
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.grid(True)
plt.show()