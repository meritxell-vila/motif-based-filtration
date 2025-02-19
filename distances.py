import networkx as nx
import numpy as np

def generate_shortest_path_distance_matrix_new(G: nx.Graph):
    # Ensure node labels are sequential from 0 to len(G)-1
    nodes = list(G.nodes)  
    node_index = {node: i for i, node in enumerate(nodes)} 

    distance_matrix = np.ones((len(nodes), len(nodes))) * np.inf
    shortest_paths = dict(nx.all_pairs_shortest_path_length(G))
    for node in nodes:
        for target_node, distance in shortest_paths.get(node, {}).items():
            distance_matrix[node_index[node], node_index[target_node]] = distance
    return distance_matrix


def generate_shortest_path_distance_matrix(G: nx.Graph):
    shortest_paths = nx.all_pairs_shortest_path_length(G)
    nodes = G.nodes
    distance_matrix = np.ones((len(nodes), len(nodes))) * np.inf
    for node, shortest_paths_to_node in shortest_paths:
        for target_node, distance in shortest_paths_to_node.items():
            distance_matrix[node, target_node] = distance
    return distance_matrix