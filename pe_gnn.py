"""Two-layer GCN/GraphSAGE/GAT/Transformer models, with optional PE and Moran.
This code is based on Klenner et al., Positional encoder graph neural networks
for geographic data. Proceedings of the 26th International Conference on Artificial
Intelligence and Statistics 206, 1379–1389 (2023)."""


import argparse
import json
import random
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, TransformerConv, knn_graph

from preprocessing import save_predictions, summarize


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(value="auto"):
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(value)


def encode_positions(coordinates):
    frequencies = 1 / np.exp(np.linspace(np.log(1e-6), np.log(360.), 16))
    angles = np.asarray(coordinates)[:, :, None] * frequencies[None, None, :]
    return torch.tensor(np.stack([np.sin(angles), np.cos(angles)], axis=-1)
                        .reshape(len(coordinates), 64), dtype=torch.float32)


def build_graph(coordinates, device="cpu", k=5):
    """Euclidean degree-space k-NN, haversine edge weights.
    """
    coords = torch.tensor(np.array(coordinates, copy=True), dtype=torch.float64)
    edges = knn_graph(coords, k=k, loop=False, flow="source_to_target")
    source, destination = edges
    rad = torch.deg2rad(coords)
    lat1, lon1 = rad[source, 0], rad[source, 1]
    lat2, lon2 = rad[destination, 0], rad[destination, 1]
    a = torch.sin((lat2-lat1)/2)**2 + torch.cos(lat1)*torch.cos(lat2)*torch.sin((lon2-lon1)/2)**2
    distances = 6371 * 2 * torch.asin(torch.sqrt(a.clamp(0, 1)))
    spread = distances.max()-distances.min()
    weights = (distances.max()-distances)/spread if spread.item() > 0 else torch.ones_like(distances)
    return edges.to(device), weights.to(device=device, dtype=torch.float32)


class GraphModel(nn.Module):
    def __init__(self, n_features=44, family="gcn", pe=True, heads=1, moran=False):
        super().__init__()
        if family not in ["gcn", "sage", "gat", "transformer"] or heads not in [1, 2, 4]:
            raise ValueError("Unsupported graph family or head count.")
        self.config = dict(n_features=n_features, family=family, pe=pe, heads=heads, moran=moran)
        self.family, self.pe, self.moran = family, pe, moran
        pe_size = (64 if family == "gcn" else 16) if pe else 0
        if pe:
            self.spenc = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(.5))
            nn.init.xavier_uniform_(self.spenc[0].weight)
            self.dec = nn.Sequential(nn.Linear(64, 32), nn.Tanh(), nn.Linear(32, 16),
                                     nn.Tanh(), nn.Linear(16, pe_size))
        def convolution(inputs):
            if family == "gcn":
                return GCNConv(inputs, 32, cached=False)
            if family == "sage":
                return SAGEConv(inputs, 32, aggr="mean")
            if family == "gat":
                return GATConv(inputs, 32//heads, heads=heads, concat=True, dropout=0.)
            return TransformerConv(inputs, 32//heads, heads=heads, concat=True,
                                   dropout=0., beta=False, root_weight=True)
        self.conv1, self.conv2 = convolution(n_features+pe_size), convolution(32)
        self.dropout = nn.Dropout(.5)
        self.fc = nn.Linear(32, 1)
        if moran:
            self.fc_morans = nn.Linear(32, 1)

    def forward(self, x, position, edge_index, edge_weight):
        if self.pe:
            x = torch.cat([x, self.dec(self.spenc(position))], dim=1)
        for layer in [self.conv1, self.conv2]:
            x = layer(x, edge_index, edge_weight) if self.family == "gcn" else layer(x, edge_index)
            x = self.dropout(torch.relu(x))
        soc = self.fc(x).squeeze(-1)
        return (soc, self.fc_morans(x).squeeze(-1)) if self.moran else soc


def train_model(model, data, seed=1, epochs=1000, device="auto", weight=0.):
    """Shared training loop for graph models and MLP; select using validation SOC only."""
    from moran import combined_loss
    device = select_device(device)
    model = model.to(device)
    graph = isinstance(model, GraphModel)
    X = {key: torch.tensor(part["X"], dtype=torch.float32) for key, part in
         ((key, data[key]) for key in ["train", "val", "test"])}
    y = {key: torch.tensor(data[key]["y"], dtype=torch.float32) for key in X}
    position = {key: encode_positions(data[key]["coordinates"]) for key in X} if graph else {}
    edges = {key: build_graph(data[key]["coordinates"], device) for key in
             ["val", "test"]} if graph else {}
    loader = DataLoader(TensorDataset(torch.arange(len(X["train"]))), batch_size=512,
                        shuffle=True, drop_last=True)
    if not len(loader):
        raise ValueError("At least 512 training observations are required.")
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    def predict(subset):
        if graph:
            result = model(X[subset].to(device), position[subset].to(device), *edges[subset])
            return result[0] if model.moran else result
        return model(X[subset].to(device))
    best_loss, state, selected_epoch = np.inf, None, None
    history = []
    start = perf_counter()
    for epoch in range(1, epochs+1):
        model.train()
        totals = np.zeros(3)
        for (idx,) in loader:
            batch_y = y["train"][idx].to(device)
            optimizer.zero_grad()
            if graph:
                edge = build_graph(data["train"]["coordinates"][idx.numpy()], device)
                output = model(X["train"][idx].to(device), position["train"][idx].to(device), *edge)
            else:
                output = model(X["train"][idx].to(device))
            if graph and model.moran:
                loss, soc_loss, auxiliary = combined_loss(output[0], batch_y, output[1], edge[0], weight)
            else:
                loss = soc_loss = nn.functional.mse_loss(output, batch_y)
                auxiliary = torch.tensor(0.)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            totals += [soc_loss.item()*len(idx), auxiliary.item()*len(idx), len(idx)]
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.mse_loss(predict("val"), y["val"].to(device)).item()
        if not np.isfinite(val_loss):
            raise RuntimeError("Non-finite validation loss.")
        history.append(dict(Epoch=epoch, Training_SOC_MSE=totals[0]/totals[2],
                            Training_Moran_MSE=totals[1]/totals[2], Validation_SOC_MSE=val_loss))
        if val_loss < best_loss:
            best_loss, selected_epoch = val_loss, epoch
            state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        if epoch == 1 or epoch % 200 == 0:
            print(f"Seed {seed} epoch {epoch}: validation log-MSE {val_loss:.4f}", flush=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    with torch.no_grad():
        predictions = {key: predict(key).cpu().numpy() for key in ["val", "test"]}
    details = dict(seed=seed, epochs=epochs, selected_epoch=selected_epoch,
                   validation_log_mse=best_loss, training_seconds=perf_counter()-start,
                   batch_size=512, drop_last=True, optimizer="Adam", learning_rate=.001,
                   early_stopping=False, device=str(device), moran_weight=weight)
    return state, predictions, pd.DataFrame(history), details


def run_graph_experiments(data_path, output, seeds, epochs, device, moran_weights=None):
    data = joblib.load(data_path)
    for family in ["gcn", "sage", "gat", "transformer"]:
        for pe in ([True] if moran_weights is not None else [False, True]):
            for heads in ([1, 2, 4] if family in ["gat", "transformer"] else [1]):
                for weight in (moran_weights if moran_weights is not None else [0.]):
                    name = ("PE-" if pe else "") + family
                    if family in ["gat", "transformer"]:
                        name += f"_heads{heads}"
                    if moran_weights is not None:
                        name += f"_Moran{weight:g}"
                    for seed in seeds:
                        set_seed(seed)
                        config = dict(n_features=len(data["features"]), family=family, pe=pe,
                                      heads=heads, moran=moran_weights is not None)
                        state, predictions, history, details = train_model(
                            GraphModel(**config), data, seed, epochs, device, weight)
                        folder = Path(output) / f"{name}_seed{seed}"
                        save_predictions(folder, name, seed, data, predictions)
                        torch.save(dict(state_dict=state, config=config, details=details), folder/"model.pt")
                        history.to_csv(folder/"training.csv", index=False)
                        (folder/"settings.json").write_text(json.dumps(details, indent=2))
                        print(name, details)
    summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/data.joblib")
    parser.add_argument("--output", default="results/graphs")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_graph_experiments(args.data, args.output, args.seeds, args.epochs, args.device)
