from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from queue import Queue


class GraphVisualization:

    def __init__(self):
        self.visual = []
        self.nodes = []

    def add_edge(self, a, b):
        temp = [a, b]
        self.visual.append(temp)

    def add_node(self, a):
        self.nodes.append(a)

    def visualize(self):
        G = nx.Graph()
        for node in self.nodes:
            G.add_node(node)
        G.add_edges_from(self.visual)
        nx.draw_networkx(G)
        figure = plt.gcf()
        figure.set_size_inches(12, 8)
        plt.savefig('foo.png')


def generate_adj_matrix(vert_n, edge_n):
    adj_matrix = np.zeros(shape=(vert_n, vert_n))
    all_edges = np.zeros(int(vert_n * (vert_n - 1) / 2))
    all_edges[:edge_n] = 1
    np.random.shuffle(all_edges)
    k = 0
    for i in range(vert_n - 1):
        for j in range(i + 1, vert_n):
            adj_matrix[i][j] = adj_matrix[j][i] = all_edges[k]
            k += 1
    return adj_matrix


def convert_to_adj_dict(adj_matrix):
    adj_dict = defaultdict(list)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] == 1:
                adj_dict[i].append(j)
    return adj_dict


def visualize_graph(adj_list, vert_num):
    G = GraphVisualization()

    for i, cur_list in adj_list.items():
        for elem in cur_list:
            G.add_edge(i, elem)
    for i in range(vert_num):
        G.add_node(i)
    G.visualize()


def dfs(adj_list, used, cur_v, component):
    used[cur_v] = True
    component.append(cur_v)
    for elem in adj_list[cur_v]:
        if not used[elem]:
            dfs(adj_list, used, elem, component)


def find_connected_components(adj_list, n):
    used = [False for _ in range(n)]
    component = []
    for i in range(n):
        if not used[i]:
            dfs(adj_list, used, i, component)
            print(component)
            component.clear()


def find_min_way(adj_list, from_vert, to_vert, vert_num):
    visited = [False for _ in range(vert_num)]
    q = Queue()
    q.put([from_vert])
    while not q.empty():
        path = q.get()
        cur_vert = path[-1]
        if cur_vert not in visited:
            for neighbour in adj_list[cur_vert]:
                new_path = list(path)
                new_path.append(neighbour)
                q.put(new_path)

                if neighbour == to_vert:
                    print(new_path)
                    return
            visited[cur_vert] = True


if __name__ == '__main__':
    vert_number = 100
    edge_number = 200
    matrix = generate_adj_matrix(vert_number, edge_number)
    print(matrix[0])
    adjs = convert_to_adj_dict(matrix)
    print(adjs[0])
    visualize_graph(adjs, vert_number)
    find_connected_components(adjs, vert_number)
    find_min_way(adjs, 2, 10, 100)
