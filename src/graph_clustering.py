import networkx as nx
import community.community_louvain as community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import modularity
import matplotlib.pyplot as plt
import os

datasets = {
    "karate": "../dataset/data-subset/karate.gml",
    "football": "../dataset/data-subset/football.gml",
    "polblogs": "../dataset/data-subset/polblogs.gml",
    "polbooks": "../dataset/data-subset/polbooks.gml",
    "strike": "../dataset/data-subset/strike.gml"
}

output_dir = "../reports"

os.makedirs(output_dir, exist_ok=True)

def visualize_graph(G, labels, filename):
    plt.figure(figsize=(10, 6))
    # Convert dictionary of labels to a list in the same order as G.nodes()
    if isinstance(labels, dict):
        node_colors = [labels[str(node)] for node in G.nodes()]
    else:
        node_colors = labels
    nx.draw(G, node_color=node_colors, with_labels=True, cmap='viridis', node_size=300, font_size=8)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

for name, path in datasets.items():
    print(f"Processing {name} dataset...")
    try:
        # Read the graph and remove duplicate edges
        G = nx.read_gml(path)
        G = nx.Graph(G)  # Convert to simple undirected graph, removing duplicates
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")

        #Louvain clustering
        louvain_partition = community_louvain.best_partition(G)
        louvain_labels = list(louvain_partition.values())
        visualize_graph(G, louvain_partition, name)

        #Spectral clustering
        adj_matrix = nx.to_numpy_array(G)
        num_clusters = len(set(louvain_labels))  # Use number of Louvain clusters
        spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
        spectral_labels = spectral_clustering.fit_predict(adj_matrix)
        visualize_graph(G, spectral_labels, f"{name}_spectral")

        #Evaluate clustering
        community_sets = [{node for node, cluster in louvain_partition.items() if cluster == i} for i in set(louvain_labels)]
        louvain_modularity = modularity(G, community_sets)
        print(f"{name} - Modularity (Louvain): {louvain_modularity}")
        
    except Exception as e:
        print(f"Error processing {name} dataset: {str(e)}")
        continue
