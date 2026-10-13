# SOC prediction

| File | Purpose |
|---|---|
| `preprocessing.py` | Load existing dataset, remove exact duplicates, create 50 km blocks, fit training-only preprocessing; shared metrics/output helpers |
| `baselines.py` | Ridge, Random Forest, Extra Trees, XGBoost, MLP, and Ridge regression kriging |
| `pe_gnn.py` | GCN, GraphSAGE, GAT and TransformerConv, with and without positional encoding; shared neural training loop |
| `moran.py` | Training-only local Moran auxiliary target and weights 0.25, 0.5, 0.75 |
| `shap_experiments.py` | Own-node and grouped neighbourhood Kernel SHAP of a specified graph checkpoint |

## Data preprocessing

The input is the already-created LUCAS 2015/2018 covariate CSV. 
The target is `ln(OC)`, with no offset; SOC must be finite and positive. 
`season` identifies survey year.

## Run

Python 3.11; install `requirements.txt`. PyG's `knn_graph` additionally needs
the optional **torch-cluster** extension compatible with your PyTorch/platform;
follow the official [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).


```sh
python preprocessing.py /path/to/existing_covariates.csv --output results
python baselines.py --data results/data.joblib --output results/baselines
python pe_gnn.py --data results/data.joblib --output results/graphs
python moran.py --data results/data.joblib --output results/graphs
```

