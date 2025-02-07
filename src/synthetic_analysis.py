import networkx as nx
import numpy as np
from cdlib import algorithms, evaluation
import matplotlib.pyplot as plt
from networkx.generators.community import LFR_benchmark_graph
from cdlib.classes.node_clustering import NodeClustering
import community.community_louvain as community_louvain
from sklearn.cluster import SpectralClustering
import os
from utils import save_results, load_results, compare_all_datasets

# Create output directory for visualizations
output_dir = "../reports/synthetic"
os.makedirs(output_dir, exist_ok=True)

# PART 3(a) - Generate synthetic datasets using LFR benchmark
def generate_lfr_graphs(n_realizations=10, mu=0.5):
    """
    Generate multiple LFR benchmark graphs with fixed parameters
    Parameters:
    - n_realizations: number of graphs to generate
    - mu: mixing parameter (default 0.5 as specified)
    """
    graphs = []
    for i in range(n_realizations):
        G = LFR_benchmark_graph(
            n=1000,              # number of nodes
            tau1=3,              # degree exponent
            tau2=1.5,            # community size exponent
            mu=mu,               # mixing parameter
            average_degree=5,     # average degree
            min_community=20,     # minimum community size
            seed=42 + i          # different seed for each realization
        )
        graphs.append(G)
    return graphs

# PART 3(b) - Qualitative evaluation
def visualize_communities(G, communities, filename):
    """
    Visualize communities in the graph
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G)
    
    # Convert communities to node colors
    node_colors = []
    for node in G.nodes():
        for idx, community in enumerate(communities):
            if node in community:
                node_colors.append(idx)
                break
                
    nx.draw(G, pos, node_color=node_colors, cmap='tab20', 
            node_size=100, with_labels=False)
    plt.savefig(os.path.join(output_dir, f"{filename}.png"))
    plt.close()

# PART 3(c) - Quantitative evaluation
def evaluate_clustering(G, communities):
    """
    Evaluate clustering using topology-based metrics
    """
    communities_obj = NodeClustering(communities, G)
    
    # Calculate metrics
    modularity = evaluation.newman_girvan_modularity(G, communities_obj).score
    conductance = evaluation.conductance(G, communities_obj).score
    
    return {
        'modularity': modularity,
        'conductance': conductance
    }

def main():
    results = {
        'synthetic': [],
        'real_classic': [],  # To be filled from graph_clustering.py results
        'real_node_label': []  # To be filled from graph_clustering.py results
    }
    
    # Generate and evaluate synthetic graphs
    print("Generating and evaluating synthetic graphs...")
    graphs = generate_lfr_graphs()
    
    for i, G in enumerate(graphs):
        # Apply both clustering algorithms
        # 1. Louvain
        louvain_communities = algorithms.louvain(G).communities
        louvain_metrics = evaluate_clustering(G, louvain_communities)
        results['synthetic'].append({
            'algorithm': 'Louvain',
            'metrics': louvain_metrics
        })
        
        # 2. Spectral Clustering
        adj_matrix = nx.to_numpy_array(G)
        n_clusters = len(louvain_communities)  # Use same number as Louvain found
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        labels = spectral.fit_predict(adj_matrix)
        
        # Convert labels to communities format
        spectral_communities = [[] for _ in range(n_clusters)]
        for node, label in enumerate(labels):
            spectral_communities[label].append(node)
        
        spectral_metrics = evaluate_clustering(G, spectral_communities)
        results['synthetic'].append({
            'algorithm': 'Spectral',
            'metrics': spectral_metrics
        })
        
        # Visualize first realization only
        if i == 0:
            visualize_communities(G, louvain_communities, f"synthetic_louvain")
            visualize_communities(G, spectral_communities, f"synthetic_spectral")
    
    # Calculate and print average metrics
    print("\nAverage Metrics for Synthetic Datasets:")
    avg_modularity = np.mean([r['metrics']['modularity'] for r in results['synthetic']])
    avg_conductance = np.mean([r['metrics']['conductance'] for r in results['synthetic']])
    print(f"Average Modularity: {avg_modularity:.4f}")
    print(f"Average Conductance: {avg_conductance:.4f}")
    
    # Save synthetic results
    save_results(results['synthetic'], 'synthetic_results.json')
    
    # Load results from real datasets
    real_results = load_results('real_datasets_results.json')
    if real_results:
        # Compare all results
        compare_all_datasets(
            real_results['real_classic'],
            real_results['real_node_label'],
            results['synthetic']
        )
    else:
        print("\nWarning: Could not load real dataset results.")
        print("Please run graph_clustering.py first.")
    
if __name__ == "__main__":
    main()