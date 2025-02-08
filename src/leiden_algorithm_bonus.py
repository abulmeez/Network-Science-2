import networkx as nx
import leidenalg as la
import igraph as ig
import numpy as np
from networkx.algorithms.community import modularity

# BONUS PART 4 IMPLEMENTATION - LEIDEN ALGORITHM
"""Paper referenced - Traag, V. A., Waltman, L., & van Eck, N. J. (2019). 
From Louvain to Leiden: Guaranteeing well-connected communities. Scientific Reports, 9, 5233."""

def leiden_algorithm(G):
    """Leiden community detection"""
    ig_graph = ig.Graph.TupleList(G.edges(), directed=False)
    partition = la.find_partition(ig_graph, la.ModularityVertexPartition)

    # Convert partitions into the dictionary format
    partition_dict = {node: comm for comm,
                      nodes in enumerate(partition) for node in nodes}
    return partition_dict

def evaluate_clustering(G, labels):
    """
    Evaluate clustering performance by Modularity.
    """
    communities = {}
    for node, cluster in labels.items():
        if cluster not in communities:
            communities[cluster] = []
        communities[cluster].append(node)

    community_list = list(communities.values())
    modularity_score = modularity(G, community_list)

    return {"modularity": modularity_score}

def run_all_datasets():
    """
    Runs Leiden community detection on all datasets.
    """
    datasets = {
        "karate": "./dataset/data-subset/karate.gml",
        "polbooks": "./dataset/data-subset/polbooks.gml",
        "citeseer": "./dataset/real-node-label/citeseer/ind.citeseer.graph",
        "cora": "./dataset/real-node-label/cora/ind.cora.graph",
        "pubmed": "./dataset/real-node-label/pubmed/ind.pubmed.graph",
        "football": "./dataset/data-subset/football.gml",
        "strike": "./dataset/data-subset/strike.gml"
    }

    results = {}

    for dataset_name, dataset_path in datasets.items():
        print(f"\nLeiden for {dataset_name}...")

        try:
            G = nx.read_gml(dataset_path, label="id")

            #Calculating Leiden
            leiden_partition = leiden_algorithm(G)

            #Evaluating results
            metrics = evaluate_clustering(G, leiden_partition)

            print(f"\n{dataset_name} Results:")
            print(f"Modularity: {metrics['modularity']:.4f}")

            results[dataset_name] = metrics
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")

    return results

if __name__ == "__main__":
    run_all_datasets()
