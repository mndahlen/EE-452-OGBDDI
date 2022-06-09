# Link prediction on OGB-DDI graph dataset

## Running
In this repo you find several handy notebooks for demos and training models:
* **GNN_laplacian_edge_features.ipynb:** Train GraphSAGE with Laplacian features as edge features.
* **GNN_original_SPD_features.ipynb:**rain GraphSAGE with SPD features as edge features.
* **graph_exploration.ipynb:** Visualize the graph and look at some key characteristics.
* **NN_laplacian_features.ipynb:** Use laplacian features as node features to predict links using a neural network.
* **NN_SPD_features.ipynb:** Use SPD features as node features to predict links using a neural network.
* **result_analytics.ipynb:** Plot results from above runs.

## Helpers
* **helpers.py:** Definitions of common classes and functions such as generation of SPD features, GraphSAGE model etc. 


## Disclaimer
This repository is a fork from https://github.com/lustoo/OGB_link_prediction, where we use their pipeline for training GraphSAGE with edge features, as well as calculating the SPD matrix. 
