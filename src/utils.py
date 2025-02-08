import json
import os
import numpy as np
import networkx as nx

# Create directories for results
RESULTS_DIR = "../results"
REPORTS_DIR = "../reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def save_results(results, filename):
    """Save results to JSON file"""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f, cls=NumpyEncoder)

def load_results(filename):
    """Load results from JSON file"""
    filepath = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return None

def compare_all_datasets(real_classic_results, real_node_label_results, synthetic_results):
    """Compare metrics across all dataset types"""
    dataset_types = {
        'Real-Classic': real_classic_results,
        'Real-Node-Label': real_node_label_results,
        'Synthetic': synthetic_results
    }
    
    print("\nComparison across all dataset types:")
    print("-" * 50)
    
    for dataset_type, results in dataset_types.items():
        if results:
            avg_modularity = sum(r['metrics']['modularity'] for r in results) / len(results)
            avg_conductance = sum(r['metrics']['conductance'] for r in results) / len(results)
            
            print(f"\n{dataset_type}:")
            print(f"Average Modularity: {avg_modularity:.4f}")
            print(f"Average Conductance: {avg_conductance:.4f}")

def analyze_network_structure(G, communities, name):
    """Detailed network analysis"""
    analysis = f"\n## Analysis of {name} Network\n\n"
    
    # Basic statistics
    analysis += "### Network Statistics\n"
    analysis += f"- Nodes: {G.number_of_nodes()}\n"
    analysis += f"- Edges: {G.number_of_edges()}\n"
    analysis += f"- Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}\n"
    analysis += f"- Density: {nx.density(G):.3f}\n"
    analysis += f"- Average clustering coefficient: {nx.average_clustering(G):.3f}\n"
    
    # Community structure
    unique_communities = set(communities.values())
    community_sizes = [sum(1 for v in communities.values() if v == c) for c in unique_communities]
    
    analysis += "\n### Community Structure\n"
    analysis += f"- Number of communities: {len(unique_communities)}\n"
    analysis += f"- Average community size: {np.mean(community_sizes):.1f}\n"
    analysis += f"- Community size range: {min(community_sizes)} to {max(community_sizes)}\n"
    
    # Inter-community connections
    analysis += "\n### Inter-community Connectivity\n"
    inter_edges = sum(1 for (u, v) in G.edges() 
                     if communities[u] != communities[v])
    analysis += f"- Inter-community edges: {inter_edges}\n"
    analysis += f"- Percentage of inter-community edges: {100*inter_edges/G.number_of_edges():.1f}%\n"
    
    return analysis

def generate_comprehensive_report(results):
    """Enhanced comprehensive report generation"""
    report = "# Network Analysis Results\n\n"
    
    # Part 1: Centrality Analysis
    if 'centrality' in results:
        report += "## Part 1: Centrality Analysis of Enron Network\n\n"
        for measure, top_nodes in results['centrality'].items():
            report += f"\n### Top 5 nodes by {measure} centrality:\n"
            for node, score in top_nodes:
                report += f"- {node}: {score:.4f}\n"
    
    # Part 2: Community Detection
    report += "\n## Part 2: Community Detection Results\n"
    
    for dataset_type in ['real_classic', 'real_node_label', 'synthetic']:
        if dataset_type in results:
            report += f"\n### {dataset_type.replace('_', ' ').title()} Datasets\n"
            
            # Calculate average metrics
            all_mod = [r['metrics']['modularity'] for r in results[dataset_type]]
            all_cond = [r['metrics']['conductance'] for r in results[dataset_type]]
            
            report += f"\nAverage Metrics:\n"
            report += f"- Modularity: {np.mean(all_mod):.3f} (±{np.std(all_mod):.3f})\n"
            report += f"- Conductance: {np.mean(all_cond):.3f} (±{np.std(all_cond):.3f})\n"
            
            # Individual results
            for result in results[dataset_type]:
                report += f"\n#### {result['dataset']} - {result['algorithm']}\n"
                for metric, value in result['metrics'].items():
                    report += f"- {metric}: {value:.4f}\n"
    
    # Save report
    with open('../reports/comprehensive_report.md', 'w') as f:
        f.write(report)