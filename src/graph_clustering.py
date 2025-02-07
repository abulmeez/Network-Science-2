import networkx as nx
import community.community_louvain as community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import modularity
import matplotlib.pyplot as plt
import os
from utils import save_results

# PART 2 - Current Status:

# ✓ Two algorithms chosen:
#   1. Louvain Method
#   2. Spectral Clustering

# (a) Algorithmic Complexity [MISSING]:
# TODO: Add complexity analysis:
# - Louvain Method: O(n log n) where n is number of nodes
# - Spectral Clustering: O(n³) where n is number of nodes

# Current datasets only include real-classic, missing real-node-label datasets
datasets = {
    "karate": "../dataset/data-subset/karate.gml",
    "football": "../dataset/data-subset/football.gml",
    "polblogs": "../dataset/data-subset/polblogs.gml",
    "polbooks": "../dataset/data-subset/polbooks.gml",
    "strike": "../dataset/data-subset/strike.gml"
}

# TODO: Add real-node-label datasets
node_label_datasets = {
    "citeseer": "../dataset/real-node-label/citeseer/",
    "cora": "../dataset/real-node-label/cora/",
    "pubmed": "../dataset/real-node-label/pubmed/"
}

output_dir = "../reports"
os.makedirs(output_dir, exist_ok=True)

# (b) Qualitative Evaluation - Visualization [PARTIAL]
def visualize_graph(G, labels, filename):
    plt.figure(figsize=(10, 6))
    if isinstance(labels, dict):
        node_colors = [labels[str(node)] for node in G.nodes()]
    else:
        node_colors = labels
    nx.draw(G, node_color=node_colors, with_labels=True, cmap='viridis', node_size=300, font_size=8)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

# (c) Quantitative Evaluation [PARTIAL]
def evaluate_clustering(G, true_labels, predicted_labels):
    """
    Evaluate clustering using both topology-based and label-dependent metrics
    """
    # Topology-based metrics
    community_sets = [{node for node, cluster in predicted_labels.items() if cluster == i} 
                     for i in set(predicted_labels.values())]
    mod = modularity(G, community_sets)
    
    # TODO: Add conductance calculation
    
    # Label-dependent metrics
    nmi = normalized_mutual_info_score(list(true_labels.values()), list(predicted_labels.values()))
    ari = adjusted_rand_score(list(true_labels.values()), list(predicted_labels.values()))
    
    return {
        'modularity': mod,
        'nmi': nmi,
        'ari': ari
    }

# Main processing loop [NEEDS UPDATE]
def main():
    results = {
        'real_classic': [],
        'real_node_label': []
    }
    
    # Process real-classic datasets
    for name, path in datasets.items():
        print(f"Processing {name} dataset...")
        try:
            # Read the graph and remove duplicate edges
            G = nx.read_gml(path)
            G = nx.Graph(G)  # Convert to simple undirected graph
            print(f"Number of nodes: {G.number_of_nodes()}")
            print(f"Number of edges: {G.number_of_edges()}")

            # Louvain clustering
            louvain_partition = community_louvain.best_partition(G)
            louvain_labels = list(louvain_partition.values())
            visualize_graph(G, louvain_partition, name)

            # Spectral clustering
            adj_matrix = nx.to_numpy_array(G)
            num_clusters = len(set(louvain_labels))
            spectral_clustering = SpectralClustering(n_clusters=num_clusters, affinity='precomputed')
            spectral_labels = spectral_clustering.fit_predict(adj_matrix)
            visualize_graph(G, spectral_labels, f"{name}_spectral")

            # Add results
            louvain_metrics = evaluate_clustering(G, louvain_partition)
            spectral_metrics = evaluate_clustering(G, spectral_labels)
            
            results['real_classic'].extend([
                {'algorithm': 'Louvain', 'dataset': name, 'metrics': louvain_metrics},
                {'algorithm': 'Spectral', 'dataset': name, 'metrics': spectral_metrics}
            ])
            
        except Exception as e:
            print(f"Error processing {name} dataset: {str(e)}")
            continue
    
    # Save results
    save_results(results, 'real_datasets_results.json')

if __name__ == "__main__":
    main()
