import collections
import math
import os

import networkx as nx
import numpy as np
import requests
from networkx import enumerate_all_cliques
from tqdm import tqdm


def create_folder_if_not_exists(path: str):
    os.makedirs(path, exist_ok=True)


def download_file(url: str, filepath: str):
    response = requests.get(url, stream=True)
    # Size in bytes
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    # Downloading using progress bar
    with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
        with open(filepath, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
                pbar.update(len(data))

    if total_size != 0 and pbar.n != total_size:
        raise RuntimeError(f"Error while downloading the file {url}")


def propagate_node_attribute_to_edges(G, attribute, pooling_fn=max):
    """Propagate a node attribute to edges.

    This function propagates a node attribute, such as the degree,
    to an edge attribute of the same name. This is done by calling
    a pooling function that condenses information of the attribute
    values of nodes incident on an edge.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Note that this graph will be changed **in place**.

    attribute : str
        Node attribute to use for the propagation procedure.

    pooling_fn : callable
        Function to pool node attribute information. Must be compatible
        with the node attribute type. The pooling function is called to
        summarize all node attributes that belong to an edge, i.e. only
        source and target node attributes.

        The pooling function must return a scalar value when provided
        with the source and target node of an edge. While other types
        of values are supported in principle, they will not result in
        graphs that are amenable to persistent homology calculations.
    """
    edge_attributes = dict()
    node_attributes = nx.get_node_attributes(G, attribute)

    for edge in G.edges(data=False):
        source, target = edge

        edge_attributes[edge] = pooling_fn(
            node_attributes[source], node_attributes[target]
        )

    nx.set_edge_attributes(G, edge_attributes, name=attribute)


def propagate_edge_attribute_to_nodes(G, attribute, pooling_fn=np.sum):
    """Propagate an edge attribute to nodes.

    This function propagates an edge attribute, such as a curvature
    measurement, to a node attribute of the same name. This is done
    by calling a pooling function that condenses information of the
    attribute values of edges incident on a node.

    Parameters
    ----------
    G : networkx.Graph
        Input graph. Note that this graph will be changed **in place**.

    attribute : str
        Edge attribute to use for the propagation procedure.

    pooling_fn : callable
        Function to pool edge attribute information. Must be compatible
        with the edge attribute type. The pooling function is called to
        summarize all edge attributes that belong to a node, i.e. *all*
        attributes belonging to incident edges.
    """
    node_attributes = collections.defaultdict(list)

    for edge in G.edges(data=True):
        source, target, data = edge

        node_attributes[source].append(data[attribute])
        node_attributes[target].append(data[attribute])

    node_attributes = {
        node: pooling_fn(values) for node, values in node_attributes.items()
    }

    nx.set_node_attributes(G, node_attributes, name=attribute)


def pad_diagrams(diagrams):
    """Pad persistence diagrams to be of the same length."""
    D = np.vstack(diagrams)

    min_d = int(np.min(D[:, 2]))
    max_d = int(np.max(D[:, 2]))

    max_features_per_dim = {}

    for diagram in diagrams:
        for dim in range(min_d, max_d + 1):
            n_features = np.sum(diagram[:, 2] == dim)

            max_features_per_dim[dim] = max(
                n_features, max_features_per_dim.get(dim, 2)
            )

    for dim, max_features in max_features_per_dim.items():
        for index, diagram in enumerate(diagrams):
            n_features = np.sum(diagram[:, 2] == dim)

            offset = max_features - n_features
            padding = np.zeros((offset, 3))

            # Ensure that values are being padded with the right
            # dimension.
            padding[:, 2] = dim

            # Somewhat inelegant: we just replace the stored diagram
            # with a padded one.
            diagrams[index] = np.vstack((diagram, padding))

    return diagrams


def to_networkx_classification_datasets(data):
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes), attr=data.x)

    for i, (u, v) in enumerate(data.edge_index.t().tolist()):
        G.add_edge(u, v)
        G[u][v]["weight"] = 1  # set weights to be 1
        try:
            G[u][v]["edge_attr"] = data.edge_attr[i]
        except:
            continue

    for i, feat_dict in G.nodes(data=True):
        try:
            feat_dict.update({"x": data.x[i]})
            feat_dict.update({"edge_attr": data.edge_attr[i]})
        except:
            continue
    G.graph["y"] = data.y

    return G


def get_clique_number(G):
    """Get the clique number of a graph."""
    cliques = list(enumerate_all_cliques(G))
    clique_number = max([len(clique) for clique in cliques])
    return clique_number


def get_max_radius_graph(G):
    c_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    radii = [nx.radius(component) for component in c_components]
    return max(radii)


def get_girth_graph(G):
    girth = nx.girth(G)
    if math.isinf(girth):
        girth = G.number_of_nodes() + 1
    return girth


def get_girth_raw_graph(G):
    girth = nx.girth(G)
    return girth


def get_max_diameter_graph(G):
    c_components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    diameters = [nx.diameter(component) for component in c_components]
    return max(diameters)
