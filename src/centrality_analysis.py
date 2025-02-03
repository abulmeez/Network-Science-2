import pandas as pd
import networkx as nx

# Read the data files with corrected paths
email_data = pd.read_csv('../dataset/email-Enron.txt', sep=' ', header=None, names=['sender', 'recipient', 'timestamp'])
adresses_data = pd.read_csv('../dataset/addresses-email-Enron.txt', sep='\t', header=None, names=['id', 'email'])

# Create a graph
G = nx.DiGraph()

# Count the number of interactions between each pair of nodes to use as weights
edge_weights = email_data.groupby(['sender', 'recipient']).size().reset_index(name='weight')

# Add weighted edges to the graph
for index, row in edge_weights.iterrows():
    G.add_edge(row['sender'], row['recipient'], weight=row['weight'])

email_map = dict(zip(adresses_data['id'], adresses_data['email']))

#calculate the centrality measures
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G, weight='weight')
closeness_centrality = nx.closeness_centrality(G)
eigenvector_centrality = nx.eigenvector_centrality(G)

def get_top_n_centrality(centrality_dict, n=5):
    return sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)[:n]

top_degree = get_top_n_centrality(degree_centrality)
top_betweenness = get_top_n_centrality(betweenness_centrality)
top_closeness = get_top_n_centrality(closeness_centrality)
top_eigenvector = get_top_n_centrality(eigenvector_centrality)

print("Top 5 nodes by degree centrality:")
for node, centrality in top_degree:
    print(f"{email_map[node]}: {centrality:.4f}")

print("\nTop 5 nodes by betweenness centrality:")
for node, centrality in top_betweenness:
    print(f"{email_map[node]}: {centrality:.4f}")

print("\nTop 5 nodes by closeness centrality:")
for node, centrality in top_closeness:
    print(f"{email_map[node]}: {centrality:.4f}")

print("\nTop 5 nodes by eigenvector centrality:")
for node, centrality in top_eigenvector:
    print(f"{email_map[node]}: {centrality:.4f}")