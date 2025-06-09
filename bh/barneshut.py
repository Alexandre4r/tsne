from tree import create_tree

def gradient(pij, ydata, y_neighbors, theta):
    t = create_tree(ydata)

    f_rep, Z = compute_f_rep()

    f_att  = compute_f_att(pij, y_neighbors, Z)

    return 4 *(f_att - f_rep / Z)


def compute_f_rep():
    return

def compute_f_att(pij_app,y_neighbors, Z):
    return
