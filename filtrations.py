"""Curvature measures for graphs."""

import warnings

import gudhi as gd
import networkx as nx
import numpy as np
import ot

from distances import generate_shortest_path_distance_matrix
from chordless_cycle_densities import compute_density_cycles

def get_filtration_types():
    return [
        "degree",
        "ollivier_ricci_curvature",
        "forman_ricci_curvature",
        "laplacian",
        "vietoris_rips",
        "density_triangles",
        "density_squares",
        "density_pentagons",
        "density_sum_cycles",
        "randic_connectivity_index",
        "harmonic_index",
        "repulsion_attraction_rule",
        "edge_betweenness_centrality",
        "random_weights"
    ]


def get_filtration_function(filtration_type: str) -> callable:
    """
    It returns the filtration function and a boolean to indicate if it is a node-level filtration.
    :param filtration_type:
    :return:
    """
    match filtration_type:
        case "degree":
            return degrees, True
        case "ollivier_ricci_curvature":
            return ollivier_ricci_curvature, False
        case "forman_ricci_curvature":
            return forman_curvature, False
        case "laplacian":
            return laplacian_eigenvalues, True
        case "vietoris_rips":
            return vietoris_rips, False
        case "density_triangles":
            return density_triangles, False
        case "density_squares":
            return density_squares, False
        case "density_pentagons":
            return density_pentagons, False
        case "density_sum_cycles":
            return density_sum_cycles, False
        case "randic_connectivity_index":
            return randic_connectivity_index, False
        case "harmonic_index":
            return harmonic_index, False
        case "repulsion_attraction_rule":
            return repulsion_attraction_rule, False
        case "edge_betweenness_centrality":
            return edge_betweenness_centrality, False
        case "random_weights":
            return random_weights, False
        case _:
            raise TypeError(f"Filtration type {filtration_type} not recognized.")


def vietoris_rips(G: nx.Graph) -> gd.RipsComplex:
    distance_matrix = generate_shortest_path_distance_matrix(G)
    cplx = gd.RipsComplex(distance_matrix=distance_matrix)
    return cplx


def ollivier_ricci_curvature(G, alpha=0.0, weight=None, prob_fn=None) -> np.ndarray:
    """Calculate Ollivier--Ricci curvature of a graph.

    This function calculates the Ollivier--Ricci curvature of a graph,
    optionally taking (positive) edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    alpha : float
        Provides laziness parameter for default probability measure. The
        measure is not compatible with a user-defined `prob_fn`. If such
        a function is set, `alpha` will be ignored.

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. If None, unweighted curvature is calculated. Notice that
        if `prob_fn` is provided, this parameter will have no effect for
        the calculation of probability measures, but it will be used for
        the calculation of shortest-path distances.

    prob_fn : callable or None
        If set, should refer to a function that calculate a probability
        measure for a given graph and a given node. This callable needs
        to satisfy the following signature:

        ``prob_fn(G, node, node_to_index)``

        Here, `G` refers to the graph, `node` to the node whose measure
        is to be calculated, and `node_to_index` to the lookup map that
        maps a node identifier to a zero-based index.

        If `prob_fn` is set, providing `alpha` will not have an effect.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    assert 0 <= alpha <= 1

    # Ensures that we can map a node to its index in the graph,
    # regardless of whether node labels or node names are being
    # used.
    node_to_index = {node: idx for idx, node in enumerate(G.nodes)}

    # This is defined inline anyway, so there is no need to follow the
    # same conventions as for the `prob_fn` parameter.
    def _make_probability_measure(node):
        values = np.zeros(len(G.nodes))
        values[node_to_index[node]] = alpha

        degree = G.degree(node, weight=weight)

        for neighbor in G.neighbors(node):

            if weight is not None:
                w = G[node][neighbor][weight]
            else:
                w = 1.0

            values[node_to_index[neighbor]] = (1 - alpha) * w / degree

        return values

    # We pre-calculate all information about the probability measure,
    # making edge calculations easier later on.
    if prob_fn is None:
        measures = list(map(_make_probability_measure, G.nodes))
    else:
        measures = list(map(lambda x: prob_fn(G, x, node_to_index), G.nodes))

    # This is the cost matrix for calculating the Ollivier--Ricci
    # curvature in practice.
    M = nx.floyd_warshall_numpy(G, weight=weight)

    curvature = []
    # we get a curvature per edge

    for edge in G.edges():
        source, target = edge

        mi = measures[node_to_index[source]]
        mj = measures[node_to_index[target]]

        distance = ot.emd2(mi, mj, M)
        curvature.append(1.0 - distance)

    return np.asarray(curvature)


def _forman_curvature_unweighted(G):
    curvature = []
    for edge in G.edges():
        source, target = edge
        source_degree = G.degree(source)
        target_degree = G.degree(target)

        source_neighbours = set(G.neighbors(source))
        target_neighbours = set(G.neighbors(target))

        n_triangles = len(source_neighbours.intersection(target_neighbours))
        curvature.append(float(4 - source_degree - target_degree + 3 * n_triangles))

    return np.asarray(curvature)



def _forman_curvature_weighted(G, weight):
    has_node_attributes = bool(nx.get_node_attributes(G, weight))

    curvature = []
    for edge in G.edges:
        source, target = edge
        source_weight, target_weight = 1.0, 1.0

        # Makes checking for duplicate edges easier below. We expect the
        # source vertex to be the (lexicographically) smaller one.
        if source > target:
            source, target = target, source

        if has_node_attributes:
            source_weight = G.nodes[source][weight]
            target_weight = G.nodes[target][weight]

        edge_weight = G[source][target][weight]

        e_curvature = source_weight / edge_weight
        e_curvature += target_weight / edge_weight

        parallel_edges = list(G.edges(source, data=weight)) + list(
            G.edges(target, data=weight)
        )

        for u, v, w in parallel_edges:
            if u > v:
                u, v = v, u

            if (u, v) == edge:
                continue
            else:
                e_curvature -= w / np.sqrt(edge_weight * w)

        e_curvature *= edge_weight
        curvature.append(float(e_curvature))

    return np.asarray(curvature)


def forman_curvature(G, weight=None):
    """Calculate Forman--Ricci curvature of a graph.

    This function calculates the Forman--Ricci curvature of a graph,
    optionally taking (positive) node and edge weights into account.

    Parameters
    ----------
    G : networkx.Graph
        Input graph

    weight : str or None
        Name of an edge attribute that is supposed to be used as an edge
        weight. Will use the same attribute to look up node weights. If
        None, unweighted curvature is calculated.

    Returns
    -------
    np.array
        An array of edge curvature values, following the ordering of
        edges of `G`.
    """
    # This calculation is much more efficient than the weighted one, so
    # we default to it in case there are no weights in the graph.
    if weight is None:
        return _forman_curvature_unweighted(G)
    else:
        return _forman_curvature_weighted(G, weight)


def degrees(G: nx.Graph) -> list:
    """Calculate degrees vector."""
    return [deg for _, deg in nx.degree(G)]


def laplacian_eigenvalues(G: nx.Graph) -> np.ndarray:
    """Calculate Laplacian and return eigenvalues."""
    with warnings.catch_warnings():
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return nx.laplacian_spectrum(G)


def _get_density_cycles(G: nx.Graph, type: str, is_rescale: bool = False) -> list:
    G_new = compute_density_cycles(G, is_rescale)
    
    weights = []
    for edge in G_new.edges:
        data = G_new.get_edge_data(*edge)

        if type == 'density_sum_cycles':
            weights.append(data['density_triangles'] + data['density_squares'] + data['density_pentagons'])
        else:
            weights.append(data[type])            

    return np.asarray(weights)


def density_triangles(G: nx.Graph) -> list:
    return _get_density_cycles(G, 'density_triangles')


def density_squares(G: nx.Graph) -> list:
    return _get_density_cycles(G, 'density_squares')


def density_pentagons(G: nx.Graph) -> list:
    return _get_density_cycles(G, 'density_pentagons')


def density_sum_cycles(G: nx.Graph) -> list:
    return _get_density_cycles(G, 'density_sum_cycles')


def randic_connectivity_index(G: nx.Graph) -> list:
    weights = []
    for u, v in G.edges:
        k_u = G.degree[u]
        k_v = G.degree[v]
        w = 1 / np.sqrt(k_v * k_u)
        weights.append(w)
    return np.asarray(weights)


def harmonic_index(G: nx.Graph) -> list:
    weights = []
    for u, v in G.edges:
        k_u = G.degree[u]
        k_v = G.degree[v]
        w = 2 / (k_v + k_u)
        weights.append(w)
    return np.asarray(weights)


def repulsion_attraction_rule(G: nx.Graph) -> list:
    # From: Nat Commu, 8(1), 1615
    weights = []
    for u, v in G.edges:
        k_u = G.degree[u]
        k_v = G.degree[v]
        CN = len(list(nx.common_neighbors(G, u, v)))
        w = (k_u + k_v + k_u * k_v) / (1 + CN)
        weights.append(w)
    return np.asarray(weights)


def edge_betweenness_centrality(G: nx.Graph) -> list:
    # From: Nat Commu, 8(1), 1615
    ebc = nx.edge_betweenness_centrality(G)
    weights = []
    for u, v in G.edges:
        w = ebc[(u, v)]
        weights.append(w)
    return np.asarray(weights)


def random_weights(G: nx.Graph) -> list:
    return np.random.uniform(0, 1, size=nx.number_of_edges(G))