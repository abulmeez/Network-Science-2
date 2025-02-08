# Network Science Assignment 2

## PART 1 - Centrality Analysis (20%)
- Implemented centrality measures: degree, betweenness, closeness, eigenvector
- Analyzed Enron email network with weighted edges
- Found top 5 most important nodes for each measure

## PART 2 - Graph Clustering (50%)

### (a) Algorithmic Complexity (10%)

#### Louvain Method
- Time Complexity: O(n log n) where n is number of nodes
- Each iteration requires O(m) operations where m is number of edges
- Usually converges in a few iterations for real networks

#### Spectral Clustering
- Time Complexity: O(n³) due to eigendecomposition
- Space Complexity: O(n²) for storing similarity matrix
- Additional O(n k²) for k-means step where k is number of clusters

### (b) Qualitative Evaluation (20%)
- Implemented visualization for all datasets
- Generated visualizations stored in reports directory
- Color-coded communities for easy interpretation

### (c) Quantitative Evaluation (20%)
Topology-based metrics:
- Modularity: measures quality of community structure
- Conductance: measures community separation

Label-dependent metrics:
- Normalized Mutual Information (NMI)
- Adjusted Rand Index (ARI)

## PART 3 - Synthetic Analysis (30%)
- Generated LFR benchmark graphs
- Parameters: n=1000, µ=0.5, tau1=3, tau2=1.5
- Compared performance across all dataset types

## Todo List

2. Documentation:
   - [ ] Document observations from visualizations
   - [ ] Create comprehensive results report

3. Testing:
   - [ ] Run test_implementation.py
   - [ ] Verify all metrics calculations
   - [ ] Test with all dataset types

4. Optional Bonus Tasks:
   - [ ] Compare with recent algorithms (last 4 years)
   - [ ] Add 3 more real-world datasets
   - [ ] Implement FARZ generator comparison

## Directory Structure