import networkx as nx
import numpy as np
from networkx.algorithms.community import modularity
from graph_clustering import louvain_clustering, spectral_clustering, evaluate_clustering
from leiden_algorithm_bonus import leiden_algorithm

def load_graph(dataset_path):
    try:
        G = nx.read_gml(dataset_path, label="id")
        return G
    except Exception as e:
        print(f"Error loading {dataset_path}: {e}")
        return None

def run_part_5():
    """Community detection for new datasets and comapre results"""

    datasets = {
        "Les Misérables": ("./dataset/new/les_miserables.gml"),
        "Internet AS": ("./dataset/new/internet_as.gml"),
        "Network Science Coauthorship": ("./dataset/new/network_science_coauthorship.gml")
    }

    for dataset_name, dataset_path in datasets.items():
        G = load_graph(dataset_path)
        if G is None:
            continue

        num_clusters = int(np.sqrt(len(G.nodes())))

        #Clustering calculations
        louvain_result = louvain_clustering(G)
        leiden_result = leiden_algorithm(G)
        spectral_result = spectral_clustering(G, num_clusters)

        #Evaluating Clustering
        louvain_score = evaluate_clustering(G, louvain_result)
        leiden_score = evaluate_clustering(G, leiden_result)
        spectral_score = evaluate_clustering(G, spectral_result)

        #results
        print(f"\n{dataset_name}")
        print(f"Louvain Modularity: {louvain_score['modularity']:.4f}")
        print(f"Leiden Modularity: {leiden_score['modularity']:.4f}")
        print(f"Spectral Modularity: {spectral_score['modularity']:.4f}")


if __name__ == "__main__":
    run_part_5()
