from bhsne.tree import create_tree
import numpy as np


def bh_gradient(pij, ydata, y_neighbors, theta2):
    t = create_tree(ydata)

    f_rep, Z = compute_f_rep(ydata, theta2, tree)
    Z_total = np.sum(Z)
    f_att = compute_f_att(pij, ydata, y_neighbors)

    return 4 * (f_att - f_rep / Z_total)


def is_far(node, sq_dist, theta2):
    if ((node.L) ** 2) / sq_dist < theta2:
        return True
    return False

def recurse_frep_Z(node, i, ydata, theta2, f_rep, Z):
    """
    Fonction récursive pour calculer Z et Z * frep pour un point i via Barnes-Hut.
    """
    if node.nb_body == 0:
        return
    diff = ydata[i] - node.com  # (2,)
    dist2 = np.sum(diff ** 2)
    if is_far(node, dist2, theta2):
        N = node.nb_body
        q = 1.0 / (1.0 + dist2)
        q2 = q ** 2
        f_rep[i] += N * q2 * diff
        Z[i]+= N * q
    elif node.isLeaf:
        if node.body is not None:
            N = node.nb_body
            if np.allclose(node.body, ydata[i]):
                N = N - 1
            diff_leaf = ydata[i] - node.body
            dist2_leaf = np.sum(diff_leaf ** 2)
            q = 1.0 / (1.0 + dist2_leaf)
            q2 = q ** 2
            f_rep[i] += N * q2 * diff_leaf
            Z[i] += N * q
    else:
        if node.child is not None:
            for child in node.child:
                recurse_frep_Z(child, i, ydata, theta2, f_rep, Z)

def compute_f_rep(ydata, theta2, tree):
    """
    Calcule Z et Z * forces répulsives via Barnes-Hut pour chaque point.
    Args:
        ydata (np.ndarray): embeddings (n_points, 2)
        tree (Tree): racine de l'arbre Barnes-Hut (doit avoir .node)
        theta (float): paramètre Barnes-Hut
    Returns:
        f_rep (np.ndarray): tableau (n_points, 2), Z * force répulsive
        Z (np.ndarray): tableau (n_points,), normalisation
    """
    n_points, n_dims = ydata.shape
    f_rep = np.zeros((n_points, n_dims))
    Z = np.zeros(n_points)

    for i in range(n_points):
        recurse_frep_Z(tree.node, i, ydata, theta2, f_rep, Z)

    return f_rep, Z


def compute_f_att(pij, ydata, y_neighbors):
    n_points, n_dims = ydata.shape
    f_att = np.zeros_like(ydata)
    for i in range(n_points):
        for j in y_neighbors[i]:
            diff = ydata[i] - ydata[j]
            norm2 = np.sum(diff ** 2)
            qijZ = 1.0 / (1.0 + norm2)
            f_att[i] += pij[i, j] * qijZ * diff
    return f_att
