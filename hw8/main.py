from random import randint

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx


def generate_adj_matrix(vert_n, edge_n):
    adj_matrix = np.zeros(shape=(vert_n, vert_n))
    all_edges = np.zeros(int(vert_n * (vert_n - 1) / 2))
    for i in range(edge_n):
        all_edges[i] = randint(1, 10)
    np.random.shuffle(all_edges)
    k = 0
    for i in range(vert_n - 1):
        for j in range(i + 1, vert_n):
            adj_matrix[i][j] = adj_matrix[j][i] = all_edges[k]
            k += 1
    return adj_matrix


def draw_graph_from_matrix(adj_matrix, min_span, name="foo.png"):
    figure = plt.gcf()
    G = nx.from_numpy_matrix(np.array(adj_matrix))
    layout = nx.spring_layout(G)
    colors = []
    edges = G.edges()
    for u, v in edges:
        if (u, v) not in min_span and (v, u) not in min_span:
            colors.append('b')
        else:
            colors.append('r')
    nx.draw_networkx(G, layout, edge_color=colors)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    figure.set_size_inches(12, 8)
    plt.savefig(name)
    plt.close(figure)


class Graph:
    def __init__(self, weighted_matrix):
        self.vertices_n = len(weighted_matrix)
        self.edges_n = int(np.count_nonzero(weighted_matrix) / 2)
        self.graph = weighted_matrix
        self.edges = set()

        for j in range(self.vertices_n):
            for k in range(self.vertices_n):
                if self.graph[j][k] != 0 and (j, k) not in self.edges:
                    self.edges.add((j, k))

        self.edges = sorted(self.edges, key=lambda e: self.graph[e[0]][e[1]])


class UF:
    def __init__(self, N):
        self._id = [i for i in range(N)]

    def connected(self, p, q):
        return self._find(p) == self._find(q)

    def union(self, p, q):
        p_root = self._find(p)
        q_root = self._find(q)
        if p_root == q_root:
            return
        self._id[p_root] = q_root

    def _find(self, p):
        while p != self._id[p]:
            p = self._id[p]
        return p


def kruskal(G: Graph):
    MST = set()

    uf = UF(G.vertices_n)
    for e in G.edges:
        u, v = e
        if uf.connected(u, v):
            continue
        uf.union(u, v)
        MST.add(e)
    return MST


def lcs(X, Y, m, n):
    L = [[0 for x in range(n + 1)] for x in range(m + 1)]

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

    index = L[m][n]

    lcs = [""] * index

    i = m
    j = n
    while i > 0 and j > 0:

        if X[i - 1] == Y[j - 1]:
            lcs[index - 1] = X[i - 1]
            i -= 1
            j -= 1
            index -= 1

        elif L[i - 1][j] > L[i][j - 1]:
            i -= 1
        else:
            j -= 1
    print(*L)

    print("LCS of " + X + " and " + Y + " is " + "".join(lcs))


def first_part():
    matrix = generate_adj_matrix(6, 10)
    g = Graph(matrix)
    res = kruskal(g)
    span_adj = [[0] * g.vertices_n for _ in range(g.vertices_n)]
    for cur in res:
        span_adj[cur[0]][cur[1]] = span_adj[cur[1]][cur[0]] = g.graph[cur[1]][cur[0]]
    draw_graph_from_matrix(matrix, res, "min_span.png")
    print(res)


def second_part():
    X = "AACB"
    Y = "ABACA"
    m = len(X)
    n = len(Y)
    lcs(X, Y, m, n)


if __name__ == '__main__':
    #first_part()
    second_part()

