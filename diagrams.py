import gudhi as gd
import networkx as nx
import numpy as np

from filtrations import ollivier_ricci_curvature, forman_curvature, vietoris_rips, density_triangles, density_squares, density_pentagons, density_sum_cycles, randic_connectivity_index, harmonic_index, repulsion_attraction_rule, edge_betweenness_centrality, random_weights
from utils import propagate_node_attribute_to_edges, propagate_edge_attribute_to_nodes
from tqdm import tqdm


def obtain_persistence_diagram(st: gd.SimplexTree, k: int):
    persistence_pairs = st.persistence(persistence_dim_max=True)

    diagrams = []

    for dimension in range(k + 1):
        diagram = [
            (c, d, dimension) for dim, (c, d) in persistence_pairs if dim == dimension
        ]

        if len(diagram) == 0:
            diagram = [(0, 0, dimension)]

        diagrams.append(diagram)

    return diagrams


def calculate_persistent_homology_from_rips_complex(cmplx: gd.RipsComplex, k: int = 3):
    st = cmplx.create_simplex_tree(max_dimension=k)
    return obtain_persistence_diagram(st, k)


def calculate_persistent_homology(G: nx.Graph, k: int = 3):
    """Calculate persistent homology of graph clique complex."""
    st = gd.SimplexTree()

    for v, w in G.nodes(data=True):
        weight = w["filtration"]
        st.insert([v], filtration=weight)

    for u, v, w in G.edges(data=True):
        weight = w["filtration"]
        st.insert([u, v], filtration=weight)

    st.make_filtration_non_decreasing()
    st.expansion(k)
    return obtain_persistence_diagram(st, k)


def get_filtration_annotated_graph_non_vr(
    graph: nx.Graph, filtration_fn: callable, node_level: bool = False
) -> nx.Graph:
    graph = graph.copy()  # We don't want to change the original graph
    filtration_values = filtration_fn(graph)
    if node_level:
        filtration_values = {v: c for v, c in zip(graph.nodes(), filtration_values)}
        nx.set_node_attributes(graph, filtration_values, "filtration")
    else:
        # Assign edge-based attribute. This is the assignment procedure
        # for when we are dealing with curvature filtrations.
        filtration_values = {e: c for e, c in zip(graph.edges(), filtration_values)}
        nx.set_edge_attributes(graph, filtration_values, "filtration")
    if node_level:
        propagate_node_attribute_to_edges(graph, "filtration")
    else:
        propagate_edge_attribute_to_nodes(graph, "filtration", pooling_fn=lambda x: -1)
    # FIX: propagate_edge_attribute_to_nodes does not propagate to nodes that are not connected to any other node.
    # As the only edge-filtrations are the Ollivier--Ricci and the Forman--Ricci curvatures, and the filtration values
    # for nodes are set to -1 by default, we can use this value to fill the missing values.
    if filtration_fn in [ollivier_ricci_curvature, forman_curvature, 
                         density_triangles, density_squares, density_pentagons, density_sum_cycles, 
                         randic_connectivity_index, harmonic_index, 
                         repulsion_attraction_rule, edge_betweenness_centrality, random_weights]:
        nx.set_node_attributes(graph, -1, "filtration")
    return graph


def get_filtration_annotated_graph(
    graph: nx.Graph, filtration_fn: callable, node_level: bool = False
) -> nx.Graph:
    if filtration_fn == vietoris_rips:
        return filtration_fn(graph)
    else:
        return get_filtration_annotated_graph_non_vr(graph, filtration_fn, node_level)


def calculate_persistence_diagram(
    graph: nx.Graph, filtration_fn: callable, k: int, node_level: bool = False
) -> np.ndarray:
    if filtration_fn == vietoris_rips:
        cmplx = filtration_fn(graph)
        diagrams = calculate_persistent_homology_from_rips_complex(cmplx, k=k)

    else:
        graph = get_filtration_annotated_graph(graph, filtration_fn, node_level)
        # Now we compute persistence diagrams
        diagrams = calculate_persistent_homology(graph, k=k)
    diagrams = np.vstack(diagrams)
    return diagrams


def calculate_persistence_diagrams(
    graphs: list[nx.Graph], filtration_fn: callable, k: int, node_level: bool = False
) -> list[np.ndarray]:
    persistence_diagrams = [
        calculate_persistence_diagram(graph, filtration_fn, k, node_level)
        for graph in graphs
    ]
    return persistence_diagrams


def are_persistence_diagrams_equal(
    pd1: np.ndarray, pd2: np.ndarray, distance_threshold: float = 1e-8
) -> bool:
    """Compare two persistence diagrams using the bottleneck distance."""
    bottleneck_distance = gd.bottleneck_distance(pd1, pd2)
    return bottleneck_distance <= distance_threshold
