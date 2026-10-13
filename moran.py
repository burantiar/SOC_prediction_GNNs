"""Training-batch local Moran target and optional auxiliary experiments.
References: Anselin, L. Local indicators of spatial association—lisa. Geographical Analysis
27, 93–115 (1995). and Klemmer, K., Safir, N. S. & Neill, D. B. Positional encoder graph neural networks
for geographic data. Proceedings of the 26th International Conference on Artificial
Intelligence and Statistics 206, 1379–1389 (2023)."""

import argparse
import torch


@torch.no_grad()
def make_moran_target(y, edge_index):
    """Binary incoming k-NN weights, no self edges, within-batch min-max scaling."""
    y = y.detach().reshape(-1)
    z = y - y.mean()
    denominator = z.square().sum()
    if denominator.item() == 0:
        return torch.zeros_like(y)
    source, destination = edge_index
    neighbour_sum = torch.zeros_like(z)
    neighbour_sum.index_add_(0, destination, z[source])
    raw = (len(y)-1) * z * neighbour_sum / denominator
    spread = raw.max() - raw.min()
    return (raw-raw.min())/spread if spread.item() > 0 else torch.zeros_like(y)


def combined_loss(soc_prediction, soc_target, moran_prediction, edge_index, weight):
    """Never call with validation/test outcomes to construct auxiliary targets."""
    soc_loss = torch.nn.functional.mse_loss(soc_prediction, soc_target)
    moran_loss = torch.nn.functional.mse_loss(
        moran_prediction, make_moran_target(soc_target, edge_index))
    return soc_loss + weight*moran_loss, soc_loss, moran_loss


if __name__ == "__main__":
    from pe_gnn import run_graph_experiments
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/data.joblib")
    parser.add_argument("--output", default="results/graphs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--weights", type=float, nargs="+", default=[.25, .5, .75])
    args = parser.parse_args()
    run_graph_experiments(args.data, args.output, args.seeds, args.epochs,
                          args.device, moran_weights=args.weights)
