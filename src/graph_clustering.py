import networkx as nx
import community.community_louvain as community_louvain
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from networkx.algorithms.community import modularity
import matplotlib.pyplot as plt
import os
from utils import save_results
import numpy as np
import pandas as pd
import logging
import pickle as pkl
import scipy.sparse as sp
from collections import defaultdict

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
    "karate": "./dataset/data-subset/karate.gml",
    "football": "./dataset/data-subset/football.gml",
    "polblogs": "./dataset/data-subset/polblogs.gml",
    "polbooks": "./dataset/data-subset/polbooks.gml",
    "strike": "./dataset/data-subset/strike.gml"
}

# TODO: Add real-node-label datasets
node_label_datasets = {
    "citeseer": {
        'graph': "./dataset/real-node-label/citeseer/ind.citeseer.graph",
        'features': "./dataset/real-node-label/citeseer/ind.citeseer.allx",
        'labels': "./dataset/real-node-label/citeseer/ind.citeseer.y"
    },
    "cora": {
        'graph': "./dataset/real-node-label/cora/ind.cora.graph",
        'features': "./dataset/real-node-label/cora/ind.cora.allx",
        'labels': "./dataset/real-node-label/cora/ind.cora.y"
    },
    "pubmed": {
        'graph': "./dataset/real-node-label/pubmed/ind.pubmed.graph",
        'features': "./dataset/real-node-label/pubmed/ind.pubmed.allx",
        'labels': "./dataset/real-node-label/pubmed/ind.pubmed.y"
    }
}

output_dir = "../reports"
os.makedirs(output_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../reports/clustering.log'),
        logging.StreamHandler()
    ]
)

# (b) Qualitative Evaluation - Visualization [PARTIAL]
def get_visualization_params(name):
    """Get dataset-specific visualization parameters"""
    params = {
        'citeseer': {
            'k': 3.0,
            'iterations': 100,
            'node_size_base': 100,
            'node_size_scale': 5
        },
        'cora': {
            'k': 2.5,
            'iterations': 100,
            'node_size_base': 80,
            'node_size_scale': 4
        },
        'pubmed': {
            'k': 4.0,
            'iterations': 150,
            'node_size_base': 60,
            'node_size_scale': 3
        }
    }
    return params.get(name, {
        'k': 2.0,
        'iterations': 50,
        'node_size_base': 50,
        'node_size_scale': 10
    })

def visualize_graph(G, labels, filename):
    """Enhanced visualization with better handling of large graphs"""
    plt.clf()
    plt.figure(figsize=(20, 20))
    
    # Handle large graphs
    if G.number_of_nodes() > 500:
        # Use sfdp_layout for large graphs
        pos = nx.spring_layout(G, k=2.0, iterations=50, seed=42)
        node_size = 20
        edge_alpha = 0.1
    else:
        pos = nx.spring_layout(G, k=1.0, iterations=100, seed=42)
        node_size = 100
        edge_alpha = 0.3
    
    # Convert labels to integers and handle missing labels
    if isinstance(labels, dict):
        labels = {node: int(comm) for node, comm in labels.items()}
    
    # Get unique communities
    communities = set(labels.values() if isinstance(labels, dict) else 
                     [labels[node] for node in G.nodes()])
    
    # Use a color map that works well for any number of communities
    colors = plt.cm.tab20(np.linspace(0, 1, len(communities)))
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=edge_alpha, edge_color='gray')
    
    # Draw nodes for each community
    for idx, comm in enumerate(communities):
        nodelist = [node for node in G.nodes() if labels.get(node, -1) == comm]
        nx.draw_networkx_nodes(G, pos,
                             nodelist=nodelist,
                             node_color=[colors[idx]],
                             node_size=node_size,
                             label=f'Community {comm}')
    
    plt.title(f"Community Structure - {filename}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.axis('off')
    
    # Save with high quality
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()

# (c) Quantitative Evaluation [PARTIAL]
def evaluate_clustering(G, partition):
    """Evaluate clustering with better error handling and output formatting"""
    try:
        # Convert partition format if needed
        if isinstance(partition, dict):
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            communities = list(communities.values())
        else:
            communities = list(partition)  # Ensure it's a list
        
        # Calculate modularity with proper community format
        try:
            node_to_comm = {}
            for comm_id, nodes in enumerate(communities):
                if isinstance(nodes, (list, set)):  # Handle both list and set formats
                    for node in nodes:
                        node_to_comm[node] = comm_id
                else:  # Handle single node case
                    node_to_comm[nodes] = comm_id
            
            # Ensure all nodes have community assignments
            for node in G.nodes():
                if node not in node_to_comm:
                    node_to_comm[node] = 0  # Assign to first community if missing
            
            if node_to_comm:  # Only calculate if we have valid communities
                mod = community_louvain.modularity(node_to_comm, G)
            else:
                mod = 0.0
        except Exception as e:
            print(f"Modularity calculation failed: {str(e)}")
            mod = 0.0
        
        # Calculate conductance
        try:
            conductances = []
            for comm in communities:
                if isinstance(comm, (list, set)) and len(comm) > 0 and len(comm) < len(G):
                    cut_size = nx.cut_size(G, set(comm))
                    volume = sum(dict(G.degree(comm)).values())
                    if volume > 0:  # Avoid division by zero
                        conductances.append(cut_size / volume)
            avg_conductance = np.mean(conductances) if conductances else 0.0
        except Exception as e:
            print(f"Conductance calculation failed: {str(e)}")
            avg_conductance = 0.0
        
        # Filter out empty communities and ensure proper format
        real_communities = []
        for comm in communities:
            if isinstance(comm, (list, set)):
                if len(comm) > 0:
                    real_communities.append(list(comm))
            else:
                real_communities.append([comm])
        
        # Format community sizes for display - limit to first 10 communities max
        comm_sizes = [len(c) for c in real_communities]
        if len(comm_sizes) > 10:
            comm_sizes_display = f"{comm_sizes[:5]} ... ({len(comm_sizes)-5} more)"
        else:
            comm_sizes_display = str(comm_sizes)
        
        metrics = {
            'modularity': mod,
            'conductance': avg_conductance,
            'num_communities': len(real_communities),
            'community_sizes': comm_sizes[:10],  # Only store first 10 sizes
            'largest_community_size': max((len(c) for c in real_communities), default=0),
            'singleton_count': sum(1 for c in real_communities if len(c) == 1)
        }
        
        # Print concise formatted output once
        print(f"Metrics: mod={mod:.3f}, cond={avg_conductance:.3f}, "
              f"communities={len(real_communities)}, sizes={comm_sizes_display}, "
              f"max_size={metrics['largest_community_size']}, "
              f"singletons={metrics['singleton_count']}")
        
        return metrics
        
    except Exception as e:
        print(f"Evaluation failed: {str(e)}")
        return {
            'modularity': 0.0,
            'conductance': 0.0,
            'num_communities': 0,
            'community_sizes': [],
            'largest_community_size': 0,
            'singleton_count': 0
        }

def read_graph_safely(path):
    """
    Safely read a GML file, handling duplicates and other issues
    """
    try:
        # First try reading with networkx's built-in function
        G = nx.read_gml(path, label='id')
    except nx.NetworkXError as e:
        if "duplicated" in str(e):
            # Read the file manually and skip duplicates
            edges = set()
            nodes = set()
            G = nx.Graph()
            
            with open(path, 'r') as f:
                lines = f.readlines()
                
            current_edge = None
            for line in lines:
                line = line.strip()
                if line.startswith('node'):
                    nodes.add(len(nodes))
                elif line.startswith('edge'):
                    current_edge = []
                elif line.startswith('source') and current_edge is not None:
                    current_edge.append(int(line.split()[1]))
                elif line.startswith('target') and current_edge is not None:
                    current_edge.append(int(line.split()[1]))
                    if tuple(sorted(current_edge)) not in edges:
                        edges.add(tuple(sorted(current_edge)))
                        G.add_edge(*current_edge)
                    current_edge = None
            
            for node in nodes:
                if node not in G:
                    G.add_node(node)
        else:
            raise e
    
    return G

def preprocess_for_spectral(G):
    """Enhanced preprocessing for spectral clustering"""
    if not nx.is_connected(G):
        print("Warning: Graph is not fully connected. Using largest connected component.")
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc).copy()
        
        # Force maximum of 10 communities for all graphs
        suggested_communities = min(10, int(np.sqrt(subgraph.number_of_nodes() / 2)))
        print(f"Suggesting {suggested_communities} communities.")
        return subgraph, suggested_communities
    
    return G, min(10, int(np.sqrt(G.number_of_nodes() / 2)))

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
            # Read the graph and handle duplicates
            G = read_graph_safely(path)
            G = nx.Graph(G)  # Convert to simple undirected graph
            
            # Remove self-loops and relabel nodes
            G.remove_edges_from(nx.selfloop_edges(G))
            G = nx.convert_node_labels_to_integers(G)
            
            print(f"Number of nodes: {G.number_of_nodes()}")
            print(f"Number of edges: {G.number_of_edges()}")

            # Louvain clustering
            louvain_partition = community_louvain.best_partition(G)
            visualize_graph(G, louvain_partition, name)

            # Spectral clustering with preprocessing
            processed_G, suggested_k = preprocess_for_spectral(G)
            adj_matrix = nx.to_numpy_array(processed_G)
            num_clusters = suggested_k if suggested_k else len(set(louvain_partition.values()))
            spectral = SpectralClustering(
                n_clusters=num_clusters,
                affinity='precomputed',
                n_init=10,  # More initialization attempts
                assign_labels='kmeans'
            )
            spectral_labels = spectral.fit_predict(adj_matrix)
            spectral_partition = {node: label for node, label in enumerate(spectral_labels)}

            # Map back to original node IDs if needed
            if processed_G != G:
                mapping = {i: node for i, node in enumerate(processed_G.nodes())}
                spectral_partition = {mapping[node]: label for node, label in spectral_partition.items()}

            # Visualize results
            visualize_graph(G, louvain_partition, f"{name}_louvain")
            visualize_graph(G, spectral_partition, f"{name}_spectral")

            # Calculate metrics
            louvain_metrics = evaluate_clustering(G, louvain_partition)
            spectral_metrics = evaluate_clustering(G, spectral_partition)

            print(f"Louvain metrics: {louvain_metrics}")
            print(f"Spectral metrics: {spectral_metrics}")
            
            # Add results
            results['real_classic'].extend([
                {'algorithm': 'Louvain', 'dataset': name, 'metrics': louvain_metrics},
                {'algorithm': 'Spectral', 'dataset': name, 'metrics': spectral_metrics}
            ])
            
            print(f"Successfully processed {name}")
            
        except Exception as e:
            print(f"Error processing {name} dataset: {str(e)}")
            import traceback
            print(traceback.format_exc())
            continue
    
    # Process real-node-label datasets
    logging.info("\nProcessing real-node-label datasets...")
    for name in node_label_datasets.keys():
        G = load_node_label_dataset(name)
        dataset_results = process_node_label_dataset(name, G)
        if dataset_results:
            results['real_node_label'].extend(dataset_results)
    
    # Save all results
    save_results(results, 'real_datasets_results.json')
    logging.info("Analysis complete. Results saved.")

# PART 2 - Real-node-label datasets processing
def load_node_label_dataset(name):
    """Load real-node-label datasets with correct file structure"""
    try:
        paths = node_label_datasets[name]
        
        # Load graph structure
        G = nx.Graph()
        with open(paths['graph'], 'rb') as f:
            graph_dict = pkl.load(f, encoding='latin1')
            # Add edges from adjacency dict
            for node, neighbors in graph_dict.items():
                G.add_edges_from((node, neighbor) for neighbor in neighbors)
        
        # Load features with multiple attempts
        try:
            features = sp.load_npz(paths['features'])
            features = features.todense()
        except:
            try:
                with open(paths['features'], 'rb') as f:
                    features = pkl.load(f, encoding='latin1')
                if sp.issparse(features):
                    features = features.todense()
            except:
                features = np.load(paths['features'], allow_pickle=True)
                if isinstance(features, np.ndarray) and features.dtype == object:
                    features = np.vstack([f.toarray() if sp.issparse(f) else f for f in features])
        
        # Load labels with better error handling
        try:
            with open(paths['labels'], 'rb') as f:
                labels_raw = pkl.load(f, encoding='latin1')
                
            # Convert labels to numpy array if needed
            if not isinstance(labels_raw, np.ndarray):
                labels_raw = np.array(labels_raw)
            
            # Handle different types of label formats
            if labels_raw.dtype == object:
                # Convert each element safely
                labels = []
                for l in labels_raw:
                    if isinstance(l, np.ndarray):
                        if l.size == 1:  # Handle single-element arrays
                            labels.append(int(l.item()))
                        else:  # Handle multi-element arrays
                            labels.append(int(l[0]))
                    else:
                        labels.append(int(l))
                labels = np.array(labels, dtype=int)
            else:
                # For numeric arrays, just convert to int
                labels = labels_raw.astype(int)
            
            # Ensure labels are 1-dimensional
            if len(labels.shape) > 1:
                labels = labels.flatten()
            
        except Exception as e:
            print(f"Failed to load labels for {name}: {str(e)}")
            print(f"Label format: {type(labels_raw)}")
            if isinstance(labels_raw, np.ndarray):
                print(f"Label shape: {labels_raw.shape}, dtype: {labels_raw.dtype}")
            return None
        
        # Add attributes to graph
        for node in G.nodes():
            if node < len(labels):
                G.nodes[node]['label'] = int(labels[node])
            if node < features.shape[0]:
                feat_array = np.array(features[node]).flatten()
                if feat_array.size > 0:
                    G.nodes[node]['features'] = feat_array
        
        print(f"Successfully loaded {name} dataset: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
        
    except Exception as e:
        print(f"Error loading {name} dataset: {str(e)}")
        print(f"Full error: ", e.__class__.__name__)
        import traceback
        print(traceback.format_exc())
        return None

# PART 2(c) - Label-dependent evaluation metrics
def evaluate_with_ground_truth(G, predicted_labels):
    """
    Evaluate clustering using label-dependent metrics (NMI and ARI)
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
    
    # Get true labels
    true_labels = nx.get_node_attributes(G, 'label')
    
    # Convert to lists in same order
    nodes = list(G.nodes())
    y_true = [true_labels[n] for n in nodes]
    y_pred = [predicted_labels[n] for n in nodes]
    
    return {
        'nmi': normalized_mutual_info_score(y_true, y_pred),
        'ari': adjusted_rand_score(y_true, y_pred)
    }

def get_topology_metrics(G, predicted_labels):
    """Calculate topology-based metrics with better handling of edge cases"""
    # Convert node labels to integers if they're strings
    node_to_community = {int(node): community for node, community in predicted_labels.items()}
    
    # Create communities list where each community is a set of nodes
    num_communities = max(node_to_community.values()) + 1
    communities = [set() for _ in range(num_communities)]
    for node, comm_id in node_to_community.items():
        communities[comm_id].add(node)
    
    # Remove empty communities
    communities = [c for c in communities if len(c) > 0]
    
    # Calculate metrics
    try:
        mod = modularity(G, communities)
    except Exception as e:
        print(f"Modularity calculation failed: {str(e)}")
        mod = 0.0
    
    try:
        conductances = []
        for comm in communities:
            if len(comm) > 1 and len(comm) < len(G) - 1:  # Skip trivial communities
                cond = nx.conductance(G, comm)
                if not np.isnan(cond):
                    conductances.append(cond)
        avg_conductance = sum(conductances) / len(conductances) if conductances else 0.0
    except Exception as e:
        print(f"Conductance calculation failed: {str(e)}")
        avg_conductance = 0.0
    
    # Only count non-trivial communities (size > 1)
    real_communities = [c for c in communities if len(c) > 1]
    
    return {
        'modularity': mod,
        'conductance': avg_conductance,
        'num_communities': len(real_communities),
        'community_sizes': [len(c) for c in real_communities],
        'largest_community_size': max(len(c) for c in communities),
        'singleton_count': sum(1 for c in communities if len(c) == 1)
    }

def process_node_label_dataset(name, G):
    """Process a single node-label dataset with comprehensive error handling"""
    try:
        if G is None:
            logging.error(f"Failed to load {name} dataset")
            return None
        
        logging.info(f"Processing {name} dataset with {G.number_of_nodes()} nodes")
        
        # Get ground truth labels
        true_labels = nx.get_node_attributes(G, 'label')
        num_classes = len(set(true_labels.values()))
        logging.info(f"Found {num_classes} ground truth classes")
        
        # Apply algorithms and collect metrics
        results = []
        
        # Louvain
        try:
            louvain_partition = community_louvain.best_partition(G)
            louvain_metrics = evaluate_clustering(G, louvain_partition)
            visualize_graph(G, louvain_partition, f"{name}_louvain")
            results.append({
                'algorithm': 'Louvain',
                'dataset': name,
                'metrics': louvain_metrics
            })
        except Exception as e:
            logging.error(f"Louvain clustering failed for {name}: {str(e)}")
        
        # Spectral
        try:
            processed_G, suggested_k = preprocess_for_spectral(G)
            adj_matrix = nx.to_numpy_array(processed_G)
            spectral = SpectralClustering(
                n_clusters=num_classes,
                affinity='precomputed',
                n_init=10
            )
            spectral_labels = spectral.fit_predict(adj_matrix)
            spectral_partition = {node: label for node, label in enumerate(spectral_labels)}
            spectral_metrics = evaluate_clustering(G, spectral_partition)
            visualize_graph(G, spectral_partition, f"{name}_spectral")
            results.append({
                'algorithm': 'Spectral',
                'dataset': name,
                'metrics': spectral_metrics
            })
        except Exception as e:
            logging.error(f"Spectral clustering failed for {name}: {str(e)}")
        
        return results
    
    except Exception as e:
        logging.error(f"Error processing {name} dataset: {str(e)}")
        return None

def process_dataset(name, G):
    """Process a single dataset and return its results"""
    try:
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        
        results = []
        
        # Louvain clustering
        try:
            louvain_partition = community_louvain.best_partition(G)
            louvain_metrics = evaluate_clustering(G, louvain_partition)
            results.append({
                'algorithm': 'Louvain',
                'dataset': name,
                'metrics': louvain_metrics
            })
            visualize_graph(G, louvain_partition, f"{name}_louvain")
        except Exception as e:
            print(f"Error in Louvain clustering: {str(e)}")
        
        # Spectral clustering
        try:
            processed_G, suggested_k = preprocess_for_spectral(G)
            adj_matrix = nx.to_numpy_array(processed_G)
            spectral = SpectralClustering(
                n_clusters=min(10, int(np.sqrt(processed_G.number_of_nodes()/2))),
                affinity='precomputed',
                n_init=10,
                assign_labels='kmeans'
            )
            spectral_labels = spectral.fit_predict(adj_matrix)
            spectral_partition = {node: label for node, label in enumerate(spectral_labels)}
            spectral_metrics = evaluate_clustering(processed_G, spectral_partition)
            results.append({
                'algorithm': 'Spectral',
                'dataset': name,
                'metrics': spectral_metrics
            })
            visualize_graph(G, spectral_partition, f"{name}_spectral")
        except Exception as e:
            print(f"Error in Spectral clustering: {str(e)}")
        
        if results:
            print(f"Successfully processed {name}\n")
        return results
        
    except Exception as e:
        print(f"Error processing dataset {name}: {str(e)}")
        return []

if __name__ == "__main__":
    main()

# PART 2 - Implementation Details:

# Algorithmic Complexity Analysis:
# 1. Louvain Method:
#    - Time Complexity: O(n log n) where n is number of nodes
#    - Space Complexity: O(m) where m is number of edges
#    - Key Operations:
#      * Initial community assignment: O(n)
#      * Modularity optimization: O(log n) iterations
#      * Community aggregation: O(m)
#    - Advantages: Fast for sparse graphs, automatically determines number of communities
#    - Limitations: Resolution limit, non-deterministic
#
# 2. Spectral Clustering:
#    - Time Complexity: O(n³) where n is number of nodes
#    - Space Complexity: O(n²)
#    - Key Operations:
#      * Laplacian matrix construction: O(m)
#      * Eigendecomposition: O(n³)
#      * k-means clustering: O(n k t) where k is number of clusters, t is iterations
#    - Advantages: Well-defined mathematical foundation, handles non-convex clusters
#    - Limitations: Computationally expensive for large graphs, requires pre-specified k

# Dataset Categories:
# 1. Real-Classic Datasets:
#    - karate: Zachary's Karate Club (34 nodes)
#    - football: American College Football (115 nodes)
#    - polblogs: Political Blogs (1,490 nodes)
#    - polbooks: Political Books (105 nodes)
#    - strike: Strike Dataset (24 nodes)
#
# 2. Real-Node-Label Datasets:
#    - citeseer: Scientific Publication Network (3,327 nodes)
#    - cora: Scientific Publication Network (2,708 nodes)
#    - pubmed: Biomedical Publication Network (19,717 nodes)
#
# 3. Synthetic Datasets:
#    - LFR Benchmark graphs with varying mixing parameter (μ)
#    - Parameters: n=250, tau1=2.5, tau2=1.5, min_degree=3, max_degree=20
#    - Community sizes: min=20, max=50

# Evaluation Metrics:
# 1. Topology-based:
#    - Modularity: Measures community structure quality (-1 to 1)
#    - Conductance: Measures inter-community connectivity (0 to 1)
#    - Community Statistics: sizes, count, singletons
#
# 2. Label-dependent (for real-node-label datasets):
#    - Normalized Mutual Information (NMI)
#    - Adjusted Rand Index (ARI)

# Visualization Approaches:
# 1. Small graphs (n ≤ 500):
#    - Spring layout with k=1.0
#    - Node size: 100
#    - Edge alpha: 0.3
#
# 2. Large graphs (n > 500):
#    - Spring layout with k=2.0
#    - Node size: 20
#    - Edge alpha: 0.1
#    - Optimized for clarity in dense networks
