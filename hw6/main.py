import time
from collections import defaultdict
import sys

import networkx as nx
import numpy as np
from random import randint

import matplotlib.pyplot as plt


INF = sys.maxsize


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


def convert_to_adj_dict(adj_matrix):
    adj_dict = defaultdict(list)
    adj_dict_oriented = defaultdict(list)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i][j] > 0:
                adj_dict[i].append((j, adj_matrix[i][j]))
                adj_dict[j].append((i, adj_matrix[i][j]))
                adj_dict_oriented[i].append((j, adj_matrix[i][j]))
    return adj_dict, adj_dict_oriented


def dijkstra(graph, vert_n, source):
    d = [INF for _ in range(vert_n)]
    used = [False for _ in range(vert_n)]

    d[source] = 0

    for i in range(vert_n):
        min_v = -1
        for j in range(vert_n):
            if not used[j] and (min_v == - 1 or d[j] < d[min_v]):
                min_v = j
        if d[min_v] == INF:
            break
        used[min_v] = True

        for elem in graph[min_v]:
            to, w = elem
            if d[min_v] + w < d[to]:
                d[to] = d[min_v] + w
    print(d)


def bellman_ford(graph, vert_n, source):
    dist = [INF for _ in range(vert_n)]
    dist[source] = 0

    for _ in range(vert_n - 1):
        for u in range(len(graph)):
            for elem in graph[u]:
                v, w = elem
                if dist[u] != INF and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w
    print(dist)

class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def a_star(maze, start, end):
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    open_list = []
    closed_list = []

    open_list.append(start_node)

    while len(open_list) > 0:

        current_node = open_list[0]
        current_index = 0
        for index, item in enumerate(open_list):
            if item.f < current_node.f:
                current_node = item
                current_index = index

        open_list.pop(current_index)
        closed_list.append(current_node)

        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]

        children = []
        for new_position in [(0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]: # Adjacent squares

            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            if maze[node_position[0]][node_position[1]] != 0:
                continue

            new_node = Node(current_node, node_position)

            children.append(new_node)

        for child in children:

            for closed_child in closed_list:
                if child == closed_child:
                    continue

            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            open_list.append(child)


def draw_graph_from_matrix(adj_matrix):
    G = nx.from_numpy_matrix(np.array(adj_matrix))
    layout = nx.spring_layout(G)
    nx.draw_networkx(G, layout)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=labels)
    figure = plt.gcf()
    figure.set_size_inches(12, 8)
    plt.savefig('foo.png')


def check_performance(alg, graph, vert_n, src):
    total_time = 0
    for _ in range(10):
        start = time.time()
        alg(graph, vert_n, src)
        total_time += time.time() - start
    print(total_time / 10)


def generate_grid(size, block_num):
    array = np.zeros((size, size))
    for i in range(block_num):
        p = (randint(0, size - 1), randint(0, size - 1))
        array[p[0]][p[1]] = 1
    return array


def do_first_part():
    vert_number, edge_number = 100, 500
    matrix = generate_adj_matrix(vert_number, edge_number)
    draw_graph_from_matrix(matrix)
    adjs, adj_oriented = convert_to_adj_dict(matrix)
    dijkstra(adjs, vert_number, 2)
    bellman_ford(adj_oriented, vert_number, 2)

    # count performance
    check_performance(dijkstra, adjs, vert_number, 2)
    check_performance(bellman_ford, adj_oriented, vert_number, 2)


def draw_grid(grid, shortest_path):
    col_labels = range(len(grid[0]))
    row_labels = range(len(grid))
    ncols, nrows = len(grid[0]), len(grid)
    for (i, j) in shortest_path:
        grid[i][j] = 2
    plt.matshow(grid)
    plt.xticks(range(ncols), col_labels)
    plt.yticks(range(nrows), row_labels)
    plt.savefig(f"grid{str(round(time.time()))}.png")


def do_second_part():
    grid_size = 10
    grid = generate_grid(grid_size, 30)
    print(grid)
    start = (0, 0)
    end = (9, 8)
    path = a_star(grid, start, end)
    print(path)
    draw_grid(grid, path)


if __name__ == '__main__':
    do_first_part()
    do_second_part()

