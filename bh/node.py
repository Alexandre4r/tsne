from config import MIN_NODE_SIZE
import numpy as np

class Node:

    def __init__(self, x, y, L):
        self.com = np.array([x + L / 2, y + L / 2], dtype=float)  # centre de masse
        self.b_array = None  # body dans la cellule
        self.x = x
        self.y = y
        self.L = L  # taille
        self.child = None  # tableau arbres fils
        self.nb_body = 0

    def __str__(self):
        if self.nb_body == 0:
            return ""
        if not (self.b_array is None):
            return f"{[body.__str__() for body in self.b_array]}"
        res = ""
        for c in self.child:
            res += f"{c.__str__()},"
        return "{" + res[:-1] + "}"

    def subdivide(self):
        """
        Ajoute quatres fils a l'arbre
        """
        new_L = (self.L) / 2
        self.child = [
            Node(self.x, self.y, new_L),
            Node(self.x + new_L, self.y, new_L),
            Node(self.x, self.y + new_L, new_L),
            Node(self.x + new_L, self.y + new_L, new_L)
        ]
        return

    def find(self, body):
        t = (self.L) / 2
        if body[1] > self.y + t:
            if body[0] > self.x + t:
                return 3
            else:
                return 2
        else:
            if body[0] > self.x + t:
                return 1
            else:
                return 0

    def insert(self, body):

        # Cas arbre vide
        if self.nb_body == 0:
            self.b_array = [body]

        # cas limite(taille de l'arbre)
        elif self.L < MIN_NODE_SIZE:
            self.b_array.append(body)

        else:
            if self.b_array != None:
                self.subdivide()
                self.child[self.find(self.b_array[0])].insert(self.b_array[0])
                self.b_array = None
            self.child[self.find(body)].insert(body)

        self.nb_body += 1
        return

    def center_of_mass(self):

        if self.b_array is not None:
            for body in self.b_array:
                self.com = (self.com[0] + body[0], self.com[1] + body[1])
            self.com = (self.com[0] / self.nb_body, self.com[1] / self.nb_body)

        elif self.child is not None:
            for child in self.child:
                child.center_of_mass()
                self.com = (self.com[0] + child.com[0], self.com[1] + child.com[1])
            self.com = (self.com[0] / self.nb_body, self.com[1] / self.nb_body)

        return
