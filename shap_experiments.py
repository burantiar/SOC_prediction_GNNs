"""Kernel SHAP  experiments"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import torch

from pe_gnn import GraphModel, build_graph, encode_positions, select_device


def explain(data_path, checkpoint_path, output, mode="own", budget=256, seed=1,
            observations=100, device="auto", node_indices=None):
    data = joblib.load(data_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = select_device(device)
    model = GraphModel(**checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    X = torch.tensor(data["test"]["X"], dtype=torch.float32, device=device)
    position = encode_positions(data["test"]["coordinates"]).to(device)
    edges, weights = build_graph(data["test"]["coordinates"], device)
    features = [i for i, name in enumerate(data["features"]) if name not in ["TH_LAT", "TH_LONG"]]
    names = [data["features"][i] for i in features]
    rng = np.random.default_rng(seed)
    background_indices = rng.choice(len(data["train"]["X"]), 20, replace=False)
    nodes = (rng.choice(len(X), min(observations, len(X)), replace=False)
             if node_indices is None else np.asarray(node_indices, dtype=int))
    if not ((nodes >= 0) & (nodes < len(X))).all():
        raise ValueError("Invalid test-subset node index.")
    background = data["train"]["X"][background_indices][:, features]
    median = torch.tensor(np.median(data["train"]["X"][:, features], axis=0),
                          dtype=X.dtype, device=device)
    source, destination = edges.cpu()
    def forward(modified):
        result = model(modified, position, edges, weights)
        return result[0] if model.moran else result
    with torch.no_grad():
        original = forward(X).cpu().numpy()
    contributions, bases = [], []
    for node in nodes:
        neighbourhood = torch.tensor([node], dtype=torch.long)
        if mode == "neighbourhood":
            for _ in range(2):
                incoming = source[torch.isin(destination, neighbourhood)]
                neighbourhood = torch.unique(torch.cat([neighbourhood, incoming]))
        neighbours = neighbourhood.to(device)
        columns = torch.tensor(features, device=device)
        actual = X[neighbours[:, None], columns[None, :]].clone()
        def predict(values):
            predictions = []
            with torch.no_grad():
                for row in np.atleast_2d(values):
                    modified = X.clone()
                    values_tensor = torch.tensor(np.array(row, copy=True), dtype=X.dtype, device=device)
                    if mode == "own":
                        modified[node, columns] = values_tensor
                    else:
                        modified[neighbours[:, None], columns[None, :]] = (
                            values_tensor*actual + (1-values_tensor)*median)
                    predictions.append(forward(modified)[node].item())
            return np.asarray(predictions)
        value = (data["test"]["X"][node, features] if mode == "own" else np.ones(len(features)))
        reference = background if mode == "own" else np.zeros((1, len(features)))
        np.testing.assert_allclose(predict(value)[0], original[node], rtol=1e-6, atol=1e-6)
        np.random.seed(seed+int(node))
        explainer = shap.KernelExplainer(predict, reference, link="identity")
        values = np.asarray(explainer.shap_values(value[None, :], nsamples=budget, l1_reg=0,
                                                  silent=True)).reshape(-1)
        base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
        np.testing.assert_allclose(base+values.sum(), original[node], rtol=1e-5, atol=1e-5)
        contributions.append(values)
        bases.append(base)
        print(f"Completed {len(contributions)}/{len(nodes)}", flush=True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    values = np.asarray(contributions)
    np.savez_compressed(output/"shap_values.npz", values=values, base_values=bases,
                        node_indices=nodes, row_indices=data["test"]["row_index"][nodes],
                        feature_names=np.asarray(names), original_prediction=original[nodes],
                        background_indices=background_indices, background=background,
                        median_reference=median.cpu().numpy())
    pd.DataFrame(dict(Feature=names, Mean_absolute_SHAP=np.abs(values).mean(axis=0),
                      Mean_signed_SHAP=values.mean(axis=0))).sort_values(
                          "Mean_absolute_SHAP", ascending=False).to_csv(output/"importance.csv", index=False)
    (output/"settings.json").write_text(json.dumps(dict(checkpoint=str(checkpoint_path),
        model_config=checkpoint["config"], mode=mode, sampling_budget=budget,
        seed=seed, observations=len(nodes), background_observations=20,
        output_scale="ln(OC)", fixed="Coordinates, positional embeddings, graph connectivity",
        neighbour_features="Fixed" if mode == "own" else "Grouped by feature within incoming two-hop neighbourhood"), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--data", default="results/data.joblib")
    parser.add_argument("--output", default="results/shap")
    parser.add_argument("--mode", choices=["own", "neighbourhood"], default="own")
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--observations", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--nodes", type=int, nargs="+")
    args = parser.parse_args()
    explain(args.data, args.checkpoint, args.output, args.mode, args.budget, args.seed,
            args.observations, args.device, args.nodes)
