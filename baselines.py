"""Ridge, RF, Extra Trees, XGBoost, MLP, and local regression kriging."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from preprocessing import save_predictions, summarize


class MLP(nn.Module):
    def __init__(self, n_features=44):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(n_features, 64), nn.ReLU(),
                                     nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.network(x).squeeze(-1)


def make_baseline(name, seed):
    if name == "ridge":
        return Ridge(alpha=1., solver="svd")
    if name == "rf":
        return RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "extra":
        return ExtraTreesRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(objective="reg:squarederror", n_estimators=1000,
                            learning_rate=.05, max_depth=6, subsample=.8,
                            colsample_bytree=.8, tree_method="hist", random_state=seed, n_jobs=-1)


def exponential_variogram(h, nugget, partial_sill, length_scale):
    return nugget + partial_sill*(1-np.exp(-h/length_scale))


class RegressionKriging:
    """Local ordinary kriging: 50 neighbours and independent nugget noise.

    Repeated observations remain separate. A spatial correction cannot distinguish
    years at identical locations. Pair sampling estimates the variogram, not SOC.
    """
    def fit(self, X, coordinates, y):
        self.regression = Ridge(alpha=1., solver="svd").fit(X, y)
        self.coordinates = np.asarray(coordinates)
        self.residuals = y-self.regression.predict(X)
        self.tree = cKDTree(self.coordinates)
        n = len(y)
        rng = np.random.default_rng(1)
        _, neighbours = self.tree.query(coordinates, k=min(21, n))
        first = np.concatenate([np.repeat(np.arange(n), neighbours.shape[1]),
                                rng.integers(0, n, 100_000)])
        second = np.concatenate([neighbours.reshape(-1), rng.integers(0, n, 100_000)])
        pairs = np.sort(np.column_stack([first, second]), axis=1)
        pairs = np.unique(pairs[pairs[:, 0] != pairs[:, 1]], axis=0)
        distance = np.linalg.norm(coordinates[pairs[:, 0]]-coordinates[pairs[:, 1]], axis=1)
        semivariance = .5*(self.residuals[pairs[:, 0]]-self.residuals[pairs[:, 1]])**2
        positive = distance > 0
        if not positive.any():
            raise ValueError("Kriging requires multiple distinct locations.")
        edges = np.unique(np.quantile(distance[positive], np.linspace(0, 1, 31)))
        bins = np.clip(np.searchsorted(edges, distance[positive], side="right")-1, 0, len(edges)-2)
        self.variogram = (pd.DataFrame(dict(bin=bins, distance_m=distance[positive],
                                           semivariance=semivariance[positive]))
                          .groupby("bin").agg(distance_m=("distance_m", "mean"),
                              semivariance=("semivariance", "mean"), pairs=("semivariance", "size")))
        h, gamma = self.variogram.distance_m.to_numpy(), self.variogram.semivariance.to_numpy()
        if (~positive).any():
            h, gamma = np.append(0., h), np.append(semivariance[~positive].mean(), gamma)
        variance = np.var(self.residuals, ddof=1)
        minimum = max(1e-10, variance*1e-8)
        nugget = max(minimum, min(gamma[0], variance*.9))
        fit = least_squares(lambda p: exponential_variogram(h, *p)-gamma,
                            [nugget, max(variance-nugget, minimum), np.median(h[h > 0])],
                            bounds=([minimum, 0., 1.], [np.inf]*3), x_scale="jac")
        if not fit.success:
            raise RuntimeError(fit.message)
        self.parameters = fit.x
        return self

    def predict(self, X, coordinates):
        nugget, partial_sill, length_scale = self.parameters
        _, neighbours = self.tree.query(coordinates, k=min(50, len(self.residuals)))
        correction = np.empty(len(X))
        for i, idx in enumerate(neighbours):
            local = self.coordinates[idx]
            covariance = partial_sill*np.exp(-cdist(local, local)/length_scale)
            covariance += nugget*np.eye(len(idx))
            target = partial_sill*np.exp(-np.linalg.norm(local-coordinates[i], axis=1)/length_scale)
            system = np.zeros((len(idx)+1, len(idx)+1))
            system[:-1, :-1], system[:-1, -1], system[-1, :-1] = covariance, 1., 1.
            weights = np.linalg.solve(system, np.append(target, 1.))[:-1]
            correction[i] = weights @ self.residuals[idx]
        return self.regression.predict(X)+correction


def run_baselines(data_path, output, seeds, models, epochs=1000, device="auto"):
    data = joblib.load(data_path)
    for name in models:
        for seed in seeds:
            start = perf_counter()
            history = None
            if name == "mlp":
                from pe_gnn import set_seed, train_model
                set_seed(seed)
                state, predictions, history, details = train_model(
                    MLP(len(data["features"])), data, seed, epochs, device)
                model = None
            else:
                model = RegressionKriging() if name == "kriging" else make_baseline(name, seed)
                if name == "kriging":
                    model.fit(data["train"]["X"], data["train"]["projected"], data["train"]["y"])
                    predictions = {key: model.predict(data[key]["X"], data[key]["projected"])
                                   for key in ["val", "test"]}
                else:
                    model.fit(data["train"]["X"], data["train"]["y"])
                    predictions = {key: model.predict(data[key]["X"]) for key in ["val", "test"]}
                details = dict(seed=seed, fitting_and_prediction_seconds=perf_counter()-start,
                               deterministic=name in ["ridge", "kriging"])
            folder = Path(output)/f"{name}_seed{seed}"
            save_predictions(folder, name, seed, data, predictions)
            if name == "mlp":
                torch.save(dict(state_dict=state, n_features=len(data["features"]), details=details),
                           folder/"model.pt")
                history.to_csv(folder/"training.csv", index=False)
            else:
                joblib.dump(model, folder/"model.joblib")
                if name == "kriging":
                    model.variogram.to_csv(folder/"variogram.csv")
                    details.update(nugget=float(model.parameters[0]), partial_sill=float(model.parameters[1]),
                                   length_scale_m=float(model.parameters[2]), neighbours=50, pair_seed=1)
            (folder/"settings.json").write_text(json.dumps(details, indent=2))
            print(f"Completed {name}, seed {seed}", flush=True)
    summarize(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="results/data.joblib")
    parser.add_argument("--output", default="results/baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--models", nargs="+", choices=["ridge", "rf", "extra", "xgb", "mlp", "kriging"],
                        default=["ridge", "rf", "extra", "xgb", "mlp", "kriging"])
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    run_baselines(args.data, args.output, args.seeds, args.models, args.epochs, args.device)
