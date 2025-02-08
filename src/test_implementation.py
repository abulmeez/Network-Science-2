import unittest
import networkx as nx
import os
import numpy as np
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent
sys.path.append(str(src_dir))

# Now import the modules
from centrality_analysis import *
from graph_clustering import *
from synthetic_analysis import *
from utils import *

class TestImplementation(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_graph = nx.karate_club_graph()
        os.makedirs("../results", exist_ok=True)
        os.makedirs("../reports", exist_ok=True)
    
    def test_centrality_analysis(self):
        """Test centrality analysis implementation"""
        # Create a simple test graph
        G = nx.DiGraph()
        G.add_edges_from([(1,2), (2,3), (3,4), (4,1)])  # Simple cycle
        
        # Test centrality measures
        metrics = {
            'degree': nx.degree_centrality(G),
            'betweenness': nx.betweenness_centrality(G),
            'closeness': nx.closeness_centrality(G),
            'eigenvector': nx.eigenvector_centrality(G)
        }
        
        # All nodes should have same centrality in a cycle
        for metric in metrics.values():
            values = list(metric.values())
            self.assertTrue(all(abs(x - values[0]) < 1e-6 for x in values))
        
        # Test Enron data loading (should handle missing data gracefully)
        result = create_email_graph()
        if result is not None:  # Only test if data is available
            G, email_map = result
            results = analyze_centrality(G, email_map)
            self.assertIsNotNone(results)
    
    def test_graph_clustering(self):
        """Test graph clustering implementation"""
        G = self.test_graph
        
        # Test Louvain
        louvain_partition = community_louvain.best_partition(G)
        self.assertTrue(len(set(louvain_partition.values())) > 1)  # Should find multiple communities
        
        try:
            # Test metrics
            metrics = evaluate_clustering(G, louvain_partition)
            self.assertIn('modularity', metrics)
            self.assertIn('conductance', metrics)
            self.assertTrue(0 <= metrics['modularity'] <= 1)  # Modularity should be between 0 and 1
            self.assertTrue(0 <= metrics['conductance'] <= 1)  # Conductance should be between 0 and 1
        except Exception as e:
            self.fail(f"Evaluation failed: {str(e)}")
    
    def test_synthetic_analysis(self):
        """Test synthetic analysis implementation"""
        # Test LFR graph generation
        G = generate_lfr_graphs(n_realizations=1)[0]
        
        # Basic graph properties
        self.assertEqual(G.number_of_nodes(), 1000)
        self.assertGreater(G.number_of_edges(), 0)
        
        # Test community detection on synthetic graph
        communities = algorithms.louvain(G).communities
        
        # Convert communities to the format expected by evaluate_clustering
        community_dict = {node: i for i, comm in enumerate(communities) for node in comm}
        
        # Test evaluation metrics
        metrics = evaluate_clustering(G, community_dict)
        
        # Verify metric properties
        self.assertTrue(0 <= metrics['modularity'] <= 1)
        self.assertTrue(0 <= metrics['conductance'] <= 1)
    
    def test_known_results(self):
        """Test against known results for Zachary's Karate Club"""
        G = self.test_graph
        
        # Known properties of karate club graph
        self.assertEqual(G.number_of_nodes(), 34)
        self.assertEqual(G.number_of_edges(), 78)
        
        # Louvain should find approximately 4 communities
        louvain_partition = community_louvain.best_partition(G)
        num_communities = len(set(louvain_partition.values()))
        self.assertGreaterEqual(num_communities, 2)
        self.assertLessEqual(num_communities, 6)
        
        # Test modularity (should be around 0.4)
        metrics = evaluate_clustering(G, louvain_partition)
        self.assertGreater(metrics['modularity'], 0.3)
        self.assertLess(metrics['modularity'], 0.5)
    
    def test_results_saving(self):
        """Test results saving and loading"""
        test_results = {
            'test': [{'metrics': {'modularity': 0.5, 'conductance': 0.3}}]
        }
        
        # Test saving
        save_results(test_results, 'test_results.json')
        self.assertTrue(os.path.exists('../results/test_results.json'))
        
        # Test loading
        loaded_results = load_results('test_results.json')
        self.assertEqual(loaded_results['test'][0]['metrics']['modularity'], 0.5)

    def test_enron_centrality_analysis(self):
        """Test Enron email network centrality analysis"""
        G, email_map = create_email_graph()
        if G is not None:  # Only test if data is available
            results = analyze_centrality(G, email_map)
            self.assertIsNotNone(results)
            
            # Check if we have all required centrality measures
            required_measures = {'degree', 'betweenness', 'closeness', 'eigenvector'}
            self.assertTrue(all(measure in results for measure in required_measures))
            
            # Check if we get top 5 for each measure
            for measure, top_nodes in results.items():
                self.assertEqual(len(top_nodes), 5)
                
            # Check if edge weights are properly considered
            edges = list(G.edges(data=True))
            self.assertTrue(any('weight' in data for _, _, data in edges))

    def test_clustering_requirements(self):
        """Test if all clustering requirements are met"""
        # a. Test algorithmic complexity documentation
        with open('README.md', 'r') as f:
            readme = f.read()
            self.assertTrue('Algorithmic Complexity' in readme)
            self.assertTrue('Louvain Method: O(' in readme)
            self.assertTrue('Spectral Clustering: O(' in readme)
        
        # b. Test visualization capabilities
        G = self.test_graph
        louvain_partition = community_louvain.best_partition(G)
        
        # Test if visualization files are created
        visualize_graph(G, louvain_partition, "test_viz")
        self.assertTrue(os.path.exists("../reports/test_viz.png"))
        
        # c. Test evaluation metrics
        metrics = evaluate_clustering(G, louvain_partition)
        required_metrics = {'modularity', 'conductance'}
        self.assertTrue(all(metric in metrics for metric in required_metrics))
        
        # Test label-dependent metrics for labeled datasets
        if hasattr(G, 'nodes') and 'label' in G.nodes[0]:
            label_metrics = evaluate_with_ground_truth(G, louvain_partition)
            required_label_metrics = {'nmi', 'ari'}
            self.assertTrue(all(metric in label_metrics for metric in required_label_metrics))

    def test_synthetic_requirements(self):
        """Test synthetic analysis requirements"""
        # a. Test LFR generation parameters
        G = generate_lfr_graphs(n_realizations=1)[0]
        self.assertEqual(G.number_of_nodes(), 1000)
        self.assertGreater(G.number_of_edges(), 0)
        
        # b. Test visualization of synthetic results
        communities = algorithms.louvain(G).communities
        visualize_communities(G, communities, "test_synthetic")
        self.assertTrue(os.path.exists("../reports/synthetic/test_synthetic.png"))
        
        # c. Test metric comparison across datasets
        results = load_results('synthetic_results.json')
        if results:
            self.assertTrue('synthetic' in results)
            metrics = results['synthetic'][0]['metrics']
            required_metrics = {'modularity', 'conductance'}
            self.assertTrue(all(metric in metrics for metric in required_metrics))

if __name__ == '__main__':
    unittest.main(verbosity=2) 