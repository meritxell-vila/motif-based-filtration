import sys
import os
import networkx as nx
import tempfile
import subprocess

res = subprocess.run(f'gfortran -O3 edgecyclesmap_print_edge_weight.for -o prog', shell=True)


def compute_density_cycles(g: nx.Graph, is_rescale: bool = False) -> nx.Graph:
    # Relabel edgelist, indices should start from 1
    mapping = dict(zip(g.nodes(), range(1, nx.number_of_nodes(g) + 1)))
    inverted_mapping = {value: key for key, value in mapping.items()}
    g_relabeled = nx.relabel_nodes(g, mapping)

    # Write tmp edgelists to file
    file0, tmp_edgelist = tempfile.mkstemp()
    nx.write_edgelist(g_relabeled, tmp_edgelist, data=False)

    file1, f1 = tempfile.mkstemp()
    file2, f2 = tempfile.mkstemp()
    file3, f3 = tempfile.mkstemp()
    file4, f4 = tempfile.mkstemp()
    file5, tmp2 = tempfile.mkstemp()
    [os.close(f) for f in [file0, file1, file2, file3, file4, file5]]
    
    subprocess.run(f"./prog {tmp_edgelist} {tmp2} {f4} {f1} {f2} {f3}", shell=True)

    g_new = nx.Graph()

    with open(f1, "r") as f_triangles, open(f2, "r") as f_squares, open(f3, "r") as f_pentagons:
        for x, y, z in zip(f_triangles, f_squares, f_pentagons):

            n1, n2, count_triangles, density_triangles = x.split()
            _, _, count_squares, density_squares = y.split()
            _, _, count_pentagons, density_pentagons = z.split()

            g_new.add_edge(inverted_mapping[int(n1)], 
                           inverted_mapping[int(n2)], 
                           count_triangles=int(count_triangles),
                           density_triangles=float(density_triangles),
                           count_squares=int(count_squares),
                           density_squares=float(density_squares),
                           count_pentagons=int(count_pentagons),
                           density_pentagons=float(density_pentagons))

    if is_rescale: # On average with rescaling we get worse results
        max_density_triangles = max(nx.get_edge_attributes(g_new, 'density_triangles').values())
        max_density_squares = max(nx.get_edge_attributes(g_new, 'density_squares').values())
        max_density_pentagons = max(nx.get_edge_attributes(g_new, 'density_pentagons').values())

        for u, v in g_new.edges():
            data = g_new.get_edge_data(u, v)
            if max_density_triangles > 0:
                g_new[u][v].update({'density_triangles': data['density_triangles'] / max_density_triangles})
            if max_density_squares > 0:
                g_new[u][v].update({'density_squares': data['density_squares'] / max_density_squares})
            if max_density_pentagons > 0:
                g_new[u][v].update({'density_pentagons': data['density_pentagons'] / max_density_pentagons})

    os.system(f'rm {f1} {f2} {f3} {f4} {tmp_edgelist} {tmp2}')
    return g_new


if __name__ == "__main__":
    input_edgelist = sys.argv[1]
    g = nx.read_edgelist(input_edgelist)
    g_new = compute_density_cycles(g)
