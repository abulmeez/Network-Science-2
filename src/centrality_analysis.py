import pandas as pd
import networkx as nx
from utils import save_results

def load_enron_data():
    """Load Enron email dataset"""
    try:
        email_data = pd.read_csv('./dataset/email-Enron.txt', sep=' ', 
                                header=None, names=['sender', 'recipient', 'timestamp'])
        addresses_data = pd.read_csv('./dataset/addresses-email-Enron.txt', 
                                   sep='\t', header=None, names=['id', 'email'])
        return email_data, addresses_data
    except FileNotFoundError:
        print("Warning: Enron dataset files not found")
        return None, None

def create_email_graph(email_data=None, addresses_data=None):
    """Create graph from email data"""
    if email_data is None or addresses_data is None:
        email_data, addresses_data = load_enron_data()
        if email_data is None:
            return None

    G = nx.DiGraph()
    
    # Count the number of interactions between each pair of nodes
    edge_weights = email_data.groupby(['sender', 'recipient']).size().reset_index(name='weight')
    
    # Add weighted edges to the graph
    for index, row in edge_weights.iterrows():
        G.add_edge(row['sender'], row['recipient'], weight=row['weight'])
    
    return G, dict(zip(addresses_data['id'], addresses_data['email']))

def get_top_n_centrality(centrality_dict, n=5):
    """Get top N nodes by centrality measure"""
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

def analyze_centrality(G=None, email_map=None):
    """Analyze centrality measures for the graph"""
    if G is None:
        G, email_map = create_email_graph()
        if G is None:
            return None
    
    results = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G, weight='weight'),
        'closeness': nx.closeness_centrality(G),
        'eigenvector': nx.eigenvector_centrality(G)
    }
    
    # Get top 5 for each measure
    top_results = {}
    for measure, values in results.items():
        top_n = get_top_n_centrality(values)
        if email_map:
            top_results[measure] = [(email_map[node], score) for node, score in top_n]
        else:
            top_results[measure] = top_n
            
    return top_results

def analyze_enron_network():
    """Complete analysis of Enron email network"""
    print("Loading Enron email network...")
    G, email_map = create_email_graph()
    
    if G is None:
        print("Error: Could not load Enron dataset")
        return
    
    print(f"\nNetwork Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2*G.number_of_edges()/G.number_of_nodes():.2f}")
    
    # Calculate multiple centrality measures
    centrality_measures = {
        'degree': nx.degree_centrality(G),
        'betweenness': nx.betweenness_centrality(G, weight='weight'),
        'closeness': nx.closeness_centrality(G, distance='weight'),
        'eigenvector': nx.eigenvector_centrality(G, weight='weight'),
        'pagerank': nx.pagerank(G, weight='weight')
    }
    
    # Get top 5 nodes for each measure
    results = {}
    print("\nTop 5 Important Individuals by Different Measures:")
    for measure, centrality_dict in centrality_measures.items():
        top_nodes = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        results[measure] = [(email_map.get(node, f"User_{node}"), score) for node, score in top_nodes]
        
        print(f"\n{measure.capitalize()} Centrality:")
        for email, score in results[measure]:
            print(f"  {email}: {score:.4f}")
    
    # Save results
    save_results({'centrality': results}, 'centrality_results.json')
    return results

if __name__ == "__main__":
    # Only run analysis when script is run directly
    G, email_map = create_email_graph()
    if G:
        results = analyze_centrality(G, email_map)
        for measure, top_nodes in results.items():
            print(f"\nTop 5 nodes by {measure} centrality:")
            for node, score in top_nodes:
                print(f"{node}: {score:.4f}")