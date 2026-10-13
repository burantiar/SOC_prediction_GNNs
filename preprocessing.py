"""Load existing covariates, freeze a spatial split, and fit training preprocessing.

No satellite features are created here. Coordinates are latitude, longitude.
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = ["LOAD DATASET COLUMNS HERE"]
METRICS = ["MSE", "RMSE", "MAE", "R2", "Bias", "MAPE (%)", "MAPE"]


def project_coordinates(coordinates):
    transform = Transformer.from_crs(4326, 3035, always_xy=True)
    east, north = transform.transform(coordinates[:, 1], coordinates[:, 0])
    return np.column_stack([east, north])


def prepare_data(csv_path, output, split_seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(csv_path)

    frame = frame.loc[:, ~frame.columns.str.startswith("Unnamed:")]
    duplicates = int(frame.duplicated().sum())
    frame = frame.drop_duplicates().reset_index(drop=True)
    X = frame[FEATURE_COLUMNS].to_numpy(dtype=float)
    coordinates = frame[["TH_LAT", "TH_LONG"]].to_numpy(dtype=float)
    oc = frame["OC"].to_numpy(dtype=float)

    projected = project_coordinates(coordinates)
    frame[["block_x", "block_y"]] = np.floor(projected / 50_000).astype(int)

    blocks = (frame.groupby(["block_x", "block_y"]).size()
              .rename("observations").reset_index()
              .sample(frac=1, random_state=split_seed).reset_index(drop=True))

    midpoint = (blocks.observations.cumsum() - blocks.observations / 2) / len(frame)
    blocks["split"] = np.select([midpoint < .70, midpoint < .85],
                                ["Training", "Validation"], default="Test")
    assignment = blocks.set_index(["block_x", "block_y"])["split"]
    frame["split"] = assignment.reindex(
        pd.MultiIndex.from_frame(frame[["block_x", "block_y"]])).to_numpy()
    assert frame.groupby(["block_x", "block_y"]).split.nunique().max() == 1
    indices = {name: np.flatnonzero(frame.split.eq(label)) for name, label in
               [("train", "Training"), ("val", "Validation"), ("test", "Test")]}
    if min(map(len, indices.values())) < 6:
        raise ValueError("Each subset needs at least six observations for k=5 graphs.")
    X[~np.isfinite(X)] = np.nan
    if np.isnan(X[indices["train"]]).all(axis=0).any():
        raise ValueError("A training predictor is entirely missing.")
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()
    scaler.fit(imputer.fit_transform(X[indices["train"]]))
    data = {"features": FEATURE_COLUMNS, "imputer": imputer, "scaler": scaler}
    
    for subset, idx in indices.items():
        data[subset] = dict(X=scaler.transform(imputer.transform(X[idx])),
                            y=np.log(oc[idx]), coordinates=coordinates[idx],
                            projected=projected[idx], row_index=idx)
    joblib.dump(data, output / "data.joblib")
    frame.insert(0, "row_index", np.arange(len(frame)))
    frame[["row_index", "POINTID", "season", "TH_LAT", "TH_LONG", "OC",
           "block_x", "block_y", "split"]].to_csv(output / "split_manifest.csv", index=False)
    counts = {key: len(value) for key, value in indices.items()}
    config = dict(features=FEATURE_COLUMNS, observations=len(frame),
                  exact_duplicates_removed=duplicates, counts=counts, split_seed=split_seed,
                  split="50 km EPSG:3035 blocks; whole-block 70/15/15 by count",
                  target="ln(OC), zero offset", imputation="Training median",
                  scaling="Training MinMaxScaler; no clipping",
                  source_file=str(Path(csv_path).resolve()))
    (output / "preprocessing.json").write_text(json.dumps(config, indent=2))
    print(counts)
    return data


def evaluate_predictions(observed_log, predicted_log):
    observed_log, predicted_log = map(lambda x: np.asarray(x).reshape(-1),
                                      (observed_log, predicted_log))
    if observed_log.shape != predicted_log.shape or not np.isfinite(predicted_log).all():
        raise ValueError("Invalid prediction shape or non-finite values.")
    rows = []
    for scale, observed, predicted in [
        ("Natural log", observed_log, predicted_log),
        ("Original (g/kg)", np.exp(observed_log), np.exp(predicted_log)),
    ]:
        if not np.isfinite(predicted).all():
            raise ValueError("Back-transformed predictions overflowed.")
        mse = mean_squared_error(observed, predicted)
        rows.append(dict(Scale=scale, MSE=mse, RMSE=np.sqrt(mse),
                         MAE=mean_absolute_error(observed, predicted),
                         R2=r2_score(observed, predicted), Bias=np.mean(predicted-observed),
                         **{"MAPE (%)": 100*np.mean(np.abs(observed-predicted)/observed)
                            if scale == "Original (g/kg)" else np.nan,
                            "MAPE (W)": np.mean(np.abs(observed-predicted)/
                                                np.maximum(np.abs(predicted), 1))
                            if scale == "Natural log" else np.nan}))
    return pd.DataFrame(rows)


def save_predictions(folder, name, seed, data, predictions):
    """Each run has its own metrics/predictions; summaries cannot double-count reruns."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows, metrics = [], []
    for subset, prediction in predictions.items():
        observed = data[subset]["y"]
        table = evaluate_predictions(observed, prediction)
        table = table.assign(Model=name, Seed=seed, Subset=subset)
        metrics.append(table)
        rows.append(pd.DataFrame(dict(Model=name, Seed=seed, Subset=subset,
                    row_index=data[subset]["row_index"],
                    latitude=data[subset]["coordinates"][:, 0],
                    longitude=data[subset]["coordinates"][:, 1],
                    observed_log=observed, predicted_log=prediction,
                    observed_soc=np.exp(observed), predicted_soc=np.exp(prediction))))
    pd.concat(metrics).to_csv(folder / "metrics.csv", index=False)
    pd.concat(rows).to_csv(folder / "predictions.csv.gz", index=False, compression="gzip")


def summarize(output):
    tables = [pd.read_csv(path) for path in Path(output).glob("*/metrics.csv")]
    if tables:
        results = pd.concat(tables, ignore_index=True)
        results.to_csv(Path(output) / "metrics_per_seed.csv", index=False)
        summary = results.groupby(["Model", "Subset", "Scale"])[METRICS].agg(["mean", "std"])
        summary.to_csv(Path(output) / "metrics_summary.csv")
        print(summary.round(4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv")
    parser.add_argument("--output", default="results")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()
    prepare_data(args.csv, args.output, args.split_seed)
