"""
Main script to run complete network analysis
"""
import os
from datetime import datetime
from centrality_analysis import analyze_enron_network
from graph_clustering import (
    main as run_clustering, 
    read_graph_safely, 
    load_node_label_dataset,
    datasets,  # Import datasets dictionary
    node_label_datasets,  # Import node_label_datasets dictionary
    process_dataset  # Import process_dataset function
)
from synthetic_analysis import generate_and_evaluate_synthetic
from generate_report import generate_full_report
import logging
from utils import save_results  # Import save_results function

def setup_logging():
    """Setup logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs('../logs', exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'../logs/analysis_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

def run_complete_analysis():
    """Run the complete analysis pipeline with better error handling"""
    setup_logging()  # Ensure logging is set up
    results = {'real_classic': [], 'real_node_label': [], 'synthetic': []}
    
    try:
        print("=" * 80)
        print("NETWORK SCIENCE - ASSIGNMENT 2: COMMUNITY DETECTION")
        print("=" * 80)
        print("\n")

        print("PART 1: ENRON EMAIL NETWORK ANALYSIS")
        print("-" * 50)
        print("Analyzing centrality measures and identifying key nodes...")
        
        # Run Enron network analysis
        try:
            analyze_enron_network()
        except Exception as e:
            print(f"Error in Enron analysis: {str(e)}")
        
        print("\nPART 2: COMMUNITY DETECTION ANALYSIS")
        print("-" * 50)
        print("\nProcessing real-classic datasets...")
        
        # Process real-classic datasets
        for name, path in datasets.items():
            try:
                G = read_graph_safely(path)
                if G is not None:
                    results['real_classic'].extend(process_dataset(name, G))
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                continue
        
        print("\nProcessing real-node-label datasets...")
        # Process real-node-label datasets
        for name in node_label_datasets.keys():
            try:
                G = load_node_label_dataset(name)
                if G is not None:
                    results['real_node_label'].extend(process_dataset(name, G))
            except Exception as e:
                print(f"Error processing {name}: {str(e)}")
                continue
        
        print("\nPART 3: SYNTHETIC NETWORK ANALYSIS")
        print("-" * 50)
        print("\nGenerating and evaluating synthetic networks...")
        print("Parameters: n=250, τ₁=2.5, τ₂=1.5, min_degree=3, max_degree=20")
        print("Community sizes: min=20, max=50")
        print("Mixing parameter (μ): 0.1 to 0.9 in steps of 0.1")
        print("Realizations per μ: 5")
        
        try:
            results['synthetic'] = generate_and_evaluate_synthetic()
        except Exception as e:
            print(f"Error in synthetic analysis: {str(e)}")
        
        print("\nPART 4: COMPARATIVE ANALYSIS")
        print("-" * 50)
        print("\nComparing results across all datasets...")
        
        # Save results
        save_results(results, os.path.join('../reports', 'clustering_results.json'))
        
        print("\nNetwork Statistics Summary:")
        print("-" * 30)
        print("Real-Classic Datasets processed:", len(results['real_classic'])//2)
        print("Real-Node-Label Datasets processed:", len(results['real_node_label'])//2)
        print("Synthetic Networks analyzed:", len(results['synthetic']))
        
        logging.info("Analysis complete. Results saved.")
        
    except Exception as e:
        logging.error(f"Error in analysis pipeline: {str(e)}")
    
    return results

def main():
    try:
        results = run_complete_analysis()
        print("\n" + "=" * 80)
        print("Analysis completed successfully!")
        print("=" * 80)
    except Exception as e:
        logging.error(f"Analysis failed: {str(e)}", exc_info=True)
        print("\n" + "=" * 80)
        print("Analysis failed. Check logs for details.")
        print("=" * 80)

if __name__ == "__main__":
    main() 