**Question 1**

centrality_analysis.py:
.venvsaifalami@saifs-MacBook-Pro src % python centrality_analysis.py
Top 5 nodes by degree centrality:
kenneth.lay@enron.com: 0.0970
sally.beck@enron.com: 0.0945
jeff.dasovich@enron.com: 0.0871
jeff.skilling@enron.com: 0.0816
tana.jones@enron.com: 0.0741

Top 5 nodes by betweenness centrality:
kenneth.lay@enron.com: 0.0536
jeff.skilling@enron.com: 0.0402
jeff.dasovich@enron.com: 0.0377
sally.beck@enron.com: 0.0370
vince.kaminski@enron.com: 0.0279

Top 5 nodes by closeness centrality:
louise.kitchen@enron.com: 0.2210
john.lavorato@enron.com: 0.2200
greg.whalley@enron.com: 0.2132
mark.taylor@enron.com: 0.2131
barry.tycholiz@enron.com: 0.2125

Top 5 nodes by eigenvector centrality:
john.lavorato@enron.com: 0.1706
tana.jones@enron.com: 0.1663
louise.kitchen@enron.com: 0.1608
sara.shackleton@enron.com: 0.1503

**Question 2**

Chosen algorithm: Louvain and spectral clustering

venvsaifalami@Mac src % python graph_clustering.py
Processing karate dataset...
Number of nodes: 34
Number of edges: 78
karate - Modularity (Louvain): 0.41880341880341876
Processing football dataset...
Error processing football dataset: edge #342 (84--3) is duplicated
Processing polblogs dataset...
Error processing polblogs dataset: edge #12757 (1047->1179) is duplicated
Processing polbooks dataset...
Number of nodes: 105
Number of edges: 441
polbooks - Modularity (Louvain): 0.5270823370920553
Processing strike dataset...
Number of nodes: 24
Number of edges: 38
strike - Modularity (Louvain): 0.5619806094182824
