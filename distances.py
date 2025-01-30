import networkx as nx
import numpy as np


def generate_shortest_path_distance_matrix(G: nx.Graph):
    shortest_paths = nx.all_pairs_shortest_path_length(G)
    nodes = G.nodes
    distance_matrix = np.ones((len(nodes), len(nodes))) * np.inf
    for node, shortest_paths_to_node in shortest_paths:
        for target_node, distance in shortest_paths_to_node.items():
            distance_matrix[node, target_node] = distance
    return distance_matrix
