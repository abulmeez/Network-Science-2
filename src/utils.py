import json
import os

# Create directories for results
RESULTS_DIR = "../results"
REPORTS_DIR = "../reports"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def save_results(results, filename):
    """Save results to JSON file"""
    filepath = os.path.join(RESULTS_DIR, filename)
    with open(filepath, 'w') as f:
        json.dump(results, f)

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