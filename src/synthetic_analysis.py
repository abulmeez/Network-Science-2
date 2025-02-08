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
try:
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory ready: {output_dir}")
except Exception as e:
    print(f"Error creating output directory: {e}")

# PART 3(a) - Generate synthetic datasets using LFR benchmark
def generate_lfr_graphs(n_realizations=10, mu=0.5):
    """Generate multiple LFR benchmark graphs with fixed parameters"""
    graphs = []
    for i in range(n_realizations):
        try:
            # More stable parameters - only specify average_degree, not min_degree
            G = LFR_benchmark_graph(
                n=500,               # Reduced number of nodes
                tau1=2.5,            # Degree exponent
                tau2=1.5,            # Community size exponent
                mu=mu,               # Mixing parameter
                average_degree=15,    # Increased average degree
                min_community=10,     # Reduced minimum community size
                max_community=50,     # Maximum community size
                max_degree=50,        # Maximum node degree
                seed=42 + i          # Different seed for each realization
            )
            if G is not None and nx.is_connected(G):
                graphs.append(G)
                print(f"Generated graph {i+1} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        except Exception as e:
            print(f"Failed to generate graph {i}: {str(e)}")
            continue
    return graphs

# PART 3(b) - Qualitative evaluation
def visualize_communities(G, partition, filename):
    """Visualize communities in the graph"""
    plt.clf()
    plt.close('all')
    
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    
    # Convert partition to list of communities
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    # Create color map
    n_communities = len(communities)
    color_map = plt.cm.get_cmap('tab20')(np.linspace(0, 1, n_communities))
    
    # Draw nodes colored by community
    for idx, (comm_id, nodes) in enumerate(communities.items()):
        nx.draw_networkx_nodes(G, pos,
                             nodelist=nodes,
                             node_color=[color_map[idx]],
                             node_size=100)
    
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    
    plt.title(f"Communities - {filename}")
    plt.axis('off')
    
    filepath = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()

# PART 3(c) - Quantitative evaluation
def evaluate_clustering(G, predicted_labels):
    """
    Evaluate clustering using topology-based metrics
    Parameters:
        G: networkx graph
        predicted_labels: dictionary of node->cluster assignments
    """
    # Convert node labels to integers if they're strings
    node_to_community = {int(node): community for node, community in predicted_labels.items()}
    
    # Create communities list where each community is a set of nodes
    num_communities = max(node_to_community.values()) + 1
    communities = [set() for _ in range(num_communities)]
    for node, comm_id in node_to_community.items():
        communities[comm_id].add(node)
    
    # Convert to list of lists for cdlib
    communities = [list(comm) for comm in communities if comm]
    
    # Create NodeClustering object
    communities_obj = NodeClustering(communities, G)
    
    # Calculate metrics
    try:
        mod = evaluation.newman_girvan_modularity(G, communities_obj).score
    except Exception as e:
        print(f"Modularity calculation failed: {str(e)}")
        mod = 0.0
    
    try:
        conductances = []
        for comm in communities:
            if len(comm) > 0 and len(comm) < len(G):
                cond = nx.conductance(G, set(comm))
                if not np.isnan(cond):
                    conductances.append(cond)
        avg_conductance = sum(conductances) / len(conductances) if conductances else 0.0
    except Exception as e:
        print(f"Conductance calculation failed: {str(e)}")
        avg_conductance = 0.0
    
    return {
        'modularity': mod,
        'conductance': avg_conductance,
        'num_communities': len(communities),
        'community_sizes': [len(c) for c in communities]
    }

def generate_and_evaluate_synthetic():
    """Comprehensive synthetic network analysis"""
    results = []
    mu_values = np.arange(0.1, 1.0, 0.1)
    n_realizations = 5
    max_attempts = 3
    
    print("Parameters: n=250, tau1=2.5, tau2=1.5, avg_degree=10\n")
    
    for mu in mu_values:
        print(f"Evaluating µ = {mu:.1f}")
        mu_results = []
        
        for i in range(n_realizations):
            success = False
            for attempt in range(max_attempts):
                try:
                    # Generate graph with more stable parameters
                    G = LFR_benchmark_graph(
                        n=250,               # Further reduced number of nodes
                        tau1=2.5,            # Degree exponent
                        tau2=1.5,            # Community size exponent
                        mu=mu,               # Mixing parameter
                        min_degree=3,        # Use min_degree instead of average_degree
                        max_degree=20,       # Reduced maximum degree
                        min_community=20,    # Increased minimum community size
                        max_community=50,    # Maximum community size
                        seed=42 + i + attempt*100  # Different seed for each attempt
                    )
                    
                    if G is not None and nx.is_connected(G):
                        # Apply both algorithms
                        louvain_partition = community_louvain.best_partition(G)
                        louvain_metrics = evaluate_clustering(G, louvain_partition)
                        
                        adj_matrix = nx.to_numpy_array(G)
                        spectral = SpectralClustering(
                            n_clusters=min(10, int(np.sqrt(G.number_of_nodes()/2))),
                            affinity='precomputed',
                            n_init=10,
                            assign_labels='kmeans'
                        )
                        spectral_labels = spectral.fit_predict(adj_matrix)
                        spectral_partition = {node: label for node, label in enumerate(spectral_labels)}
                        spectral_metrics = evaluate_clustering(G, spectral_partition)
                        
                        mu_results.append({
                            'mu': mu,
                            'realization': i,
                            'louvain_metrics': louvain_metrics,
                            'spectral_metrics': spectral_metrics
                        })
                        
                        # Visualize first successful realization only
                        if len(mu_results) == 1:
                            visualize_communities(G, louvain_partition, f"synthetic_mu{mu:.1f}_louvain")
                            visualize_communities(G, spectral_partition, f"synthetic_mu{mu:.1f}_spectral")
                        
                        success = True
                        print(f"Successfully generated graph {i+1}")
                        break
                    
                except Exception as e:
                    if attempt == max_attempts - 1:
                        print(f"Failed to generate graph {i} after {max_attempts} attempts: {str(e)}")
                    continue
            
            if not success:
                continue
        
        if mu_results:
            # Calculate average metrics
            avg_louvain_mod = np.mean([r['louvain_metrics']['modularity'] for r in mu_results])
            avg_louvain_cond = np.mean([r['louvain_metrics']['conductance'] for r in mu_results])
            avg_spectral_mod = np.mean([r['spectral_metrics']['modularity'] for r in mu_results])
            avg_spectral_cond = np.mean([r['spectral_metrics']['conductance'] for r in mu_results])
            
            print(f"\nResults for µ = {mu:.1f} (averaged over {len(mu_results)} realizations):")
            print(f"  Louvain   - Modularity: {avg_louvain_mod:.3f}, Conductance: {avg_louvain_cond:.3f}")
            print(f"  Spectral  - Modularity: {avg_spectral_mod:.3f}, Conductance: {avg_spectral_cond:.3f}")
            
            results.extend(mu_results)
    
    return results

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
        louvain_result = algorithms.louvain(G)
        # Convert communities list to node-to-community dictionary
        louvain_partition = {node: comm_id 
                           for comm_id, community in enumerate(louvain_result.communities) 
                           for node in community}
        
        louvain_metrics = evaluate_clustering(G, louvain_partition)
        results['synthetic'].append({
            'algorithm': 'Louvain',
            'metrics': louvain_metrics
        })
        
        # 2. Spectral Clustering
        adj_matrix = nx.to_numpy_array(G)
        n_clusters = len(louvain_result.communities)
        spectral = SpectralClustering(n_clusters=n_clusters, affinity='precomputed')
        spectral_labels = spectral.fit_predict(adj_matrix)
        
        # Convert labels to dictionary format
        spectral_partition = {node: label for node, label in enumerate(spectral_labels)}
        
        spectral_metrics = evaluate_clustering(G, spectral_partition)
        results['synthetic'].append({
            'algorithm': 'Spectral',
            'metrics': spectral_metrics
        })
        
        # Visualize first realization only
        if i == 0:
            print(f"\nVisualizing communities for graph {i}...")
            print(f"Number of communities in Louvain: {len(louvain_result.communities)}")
            print(f"Number of communities in Spectral: {n_clusters}")
            visualize_communities(G, louvain_result.communities, f"synthetic_louvain")
            # Convert spectral labels back to communities format for visualization
            spectral_communities = [[] for _ in range(n_clusters)]
            for node, label in spectral_partition.items():
                spectral_communities[label].append(node)
            visualize_communities(G, spectral_communities, f"synthetic_spectral")
            print("Visualization complete.")
    
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

# Synthetic Network Analysis Parameters:
# 1. LFR Benchmark Configuration:
#    - Nodes: n = 250 (reduced from 500 for stability)
#    - Power-law exponents: τ₁ = 2.5 (degrees), τ₂ = 1.5 (community sizes)
#    - Degree range: min_degree = 3, max_degree = 20
#    - Community size range: min = 20, max = 50
#    - Mixing parameter (μ): 0.1 to 0.9 in steps of 0.1
#
# 2. Generation Process:
#    - Realizations per μ: 5 (reduced from 10 for efficiency)
#    - Maximum attempts per realization: 3
#    - Seed: 42 + i + attempt*100 for reproducibility
#
# 3. Evaluation Metrics:
#    - Modularity: Community structure quality
#    - Conductance: Inter-community connectivity
#    - Community size distribution
#    - Success rate of generation attempts