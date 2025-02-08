from utils import generate_comprehensive_report, load_results
import matplotlib.pyplot as plt
import numpy as np

def generate_full_report():
    """Generate comprehensive analysis report"""
    # First collect all results
    results = {
        'centrality': load_results('centrality_results.json'),
        'real_classic': load_results('real_datasets_results.json'),
        'synthetic': load_results('synthetic_results.json')
    }
    
    report = "# Network Science Analysis Report\n\n"
    
    # Part 1: Enron Email Network Analysis
    report += "## 1. Enron Email Network Analysis\n\n"
    if results['centrality']:
        report += "### Top 5 Important Individuals by Different Centrality Measures\n\n"
        for measure, top_nodes in results['centrality'].items():
            report += f"\n#### {measure.capitalize()} Centrality\n"
            for email, score in top_nodes:
                report += f"- {email}: {score:.4f}\n"
    
    # Part 2: Community Detection
    report += "\n## 2. Community Detection Analysis\n\n"
    
    # Real Classic Networks
    report += "### 2.1 Real Classic Networks\n\n"
    for dataset_type in ['real_classic', 'real_node_label']:
        if results[dataset_type]:
            for result in results[dataset_type]:
                report += f"\n#### {result['dataset']} - {result['algorithm']}\n"
                report += "Metrics:\n"
                for metric, value in result['metrics'].items():
                    report += f"- {metric}: {value:.4f}\n"
    
    # Part 3: Synthetic Network Analysis
    report += "\n## 3. Synthetic Network Analysis\n\n"
    if results['synthetic']:
        report += "### Performance across different mixing parameters (µ)\n\n"
    
    # Save comprehensive report
    with open('../reports/comprehensive_report.md', 'w') as f:
        f.write(report)
    
    # Generate visualizations
    plot_comparative_analysis(results)
    
    return report

def plot_comparative_analysis(results):
    """Generate comparative visualizations"""
    plt.figure(figsize=(12, 6))
    
    # Plot modularity comparison
    datasets = ['real_classic', 'real_node_label', 'synthetic']
    mod_values = []
    cond_values = []
    
    for dataset in datasets:
        if dataset in results:
            mods = [r['metrics']['modularity'] for r in results[dataset]]
            conds = [r['metrics']['conductance'] for r in results[dataset]]
            mod_values.append(np.mean(mods))
            cond_values.append(np.mean(conds))
    
    plt.subplot(1, 2, 1)
    plt.bar(datasets, mod_values)
    plt.title('Average Modularity by Dataset Type')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    plt.bar(datasets, cond_values)
    plt.title('Average Conductance by Dataset Type')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('../reports/comparative_analysis.png')
    plt.close()

def add_observations(report, results):
    """Add detailed observations to the report"""
    report += "\n## Observations\n\n"
    
    # Community structure observations
    report += "### Community Structure\n"
    report += "1. Real-classic datasets show clearer community structure than synthetic\n"
    report += "2. Louvain generally finds more modular communities than Spectral\n"
    report += "3. Node-label datasets show good correlation between ground truth and detected communities\n\n"
    
    # Algorithm performance
    report += "### Algorithm Performance\n"
    report += "1. Louvain Method:\n"
    report += "   - Better modularity scores\n"
    report += "   - More computationally efficient\n"
    report += "2. Spectral Clustering:\n"
    report += "   - More balanced community sizes\n"
    report += "   - Better performance on well-separated communities\n\n"
    
    # Dataset specific insights
    report += "### Dataset Specific Insights\n"
    for dataset_type in ['real_classic', 'real_node_label', 'synthetic']:
        if dataset_type in results:
            avg_mod = np.mean([r['metrics']['modularity'] for r in results[dataset_type]])
            report += f"{dataset_type}: Average modularity = {avg_mod:.3f}\n"
    
    return report 