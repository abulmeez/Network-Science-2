"""Run complete analysis pipeline"""
from centrality_analysis import analyze_enron_network
from graph_clustering import main as run_clustering
from synthetic_analysis import main as run_synthetic
from generate_report import generate_full_report

def main():
    print("1. Running Enron Network Analysis...")
    analyze_enron_network()
    
    print("\n2. Running Graph Clustering Analysis...")
    run_clustering()
    
    print("\n3. Running Synthetic Analysis...")
    run_synthetic()
    
    print("\n4. Generating Comprehensive Report...")
    generate_full_report()

if __name__ == "__main__":
    main() 