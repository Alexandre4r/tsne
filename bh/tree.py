import numpy as np
from bhsne.node import Node


class Tree:
    def __init__(self, b_array):
        self.b_array = b_array
        self.n = b_array.shape[0]
        self.node = None
        res = find_root(b_array)
        self.x, self.y, self.L = res


def find_root(b_array):
    if b_array.shape[0] <= 1:
        return None
    xmin, ymin = np.min(b_array, axis=0)
    xmax, ymax = np.max(b_array, axis=0)
    L = max(xmax - xmin, ymax - ymin)
    return xmin, ymin, L


def create_tree(b_array):
    t = Tree(b_array)
    t.node = Node(t.x, t.y, t.L)
    for i in range(0, b_array.shape[0]):
        t.node.insert(b_array[i])
    t.node.center_of_mass()
    return t
