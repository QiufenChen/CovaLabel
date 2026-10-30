"""Evaluate HUM-L100-trained CovaLabel models on the HUM-L1000 proteome-wide R-site case."""

import os


# ======================== modify here ========================
PROJECT_DIR = "/home/qfchen/ProbePred"
PROTEIN_FEATURE_ROOT = "/data/qfchen/esm2_feature_probe"
GPU_ID = "2"
NEGATIVE_RATIO = 5
BATCH_SIZE = 256
TOP_K = [100, 500, 1000]
STRATEGIES = ["random", "cold_protein", "cold_probe"]
SEEDS = [42, 123, 2024, 3407, 8888]
# =============================================================


os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score


SOURCE_DATASET = "Human_Lysate_100uM"
CASE_DATASET = "Human_Lysate_1mM"
RATIO = f"1_{NEGATIVE_RATIO}"
MODEL_DIR = os.path.join(PROJECT_DIR, "model_src", "models", RATIO, SOURCE_DATASET)
FEATURE_DIR = os.path.join(PROJECT_DIR, "model_src", "features")
PROTEIN_DIR = os.path.join(PROTEIN_FEATURE_ROOT, "Human")
CASE_DIR = os.path.join(PROJECT_DIR, "outputs", "case_study", CASE_DATASET)
CASE_PATH = os.path.join(CASE_DIR, "proteome_candidates.csv")
OUTPUT_DIR = os.path.join(CASE_DIR, "results", RATIO)
METRIC_NAMES = ["AUROC", "AP", "AP_baseline", "AP_fold_over_baseline"] + [f"{name}@{k}" for k in TOP_K for name in ("Precision", "Recall", "EF")]

sys.path.insert(0, os.path.join(PROJECT_DIR, "model_src"))
from data import load_probe_graphs  # noqa: E402
from model import ProbeSiteModel  # noqa: E402


def load_case_dataset():
    frame = pd.read_csv(CASE_PATH, encoding="utf-8-sig").rename(columns={"Uniprot_ID": "UNIPROT_ID", "Site": "Site_type", "Postion": "Position"})
    required = {"Probe", "UNIPROT_ID", "Site_type", "Position", "Label"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing case-study columns: {sorted(missing)}")
    if frame["Probe"].nunique() != 1:
        raise ValueError("The HUM-L1000 case dataset must contain exactly one probe")
    if set(frame["Site_type"].astype(str)) != {"R"}:
        raise ValueError("The primary HUM-L1000 case dataset must contain only R candidates")
    frame["Position"] = frame["Position"].astype(int)
    frame["Label"] = frame["Label"].astype(int)
    if set(frame["Label"]) != {0, 1}:
        raise ValueError("The case dataset must contain both Label 1 and Label 0")
    if frame.duplicated(["Probe", "UNIPROT_ID", "Site_type", "Position"]).any():
        raise ValueError("Duplicate candidate sites were found")
    return frame.reset_index(drop=True)


def make_windows(embedding, positions, size=21):
    half = size // 2
    indices = positions[:, None] + torch.arange(-half, half + 1)[None, :]
    valid = (indices >= 0) & (indices < len(embedding))
    windows = embedding[indices.clamp(0, len(embedding) - 1)]
    windows[~valid] = 0
    return windows


def prepare_graph(graph, device):
    fields = ["atom_features", "reaction_roles", "adjacency", "bond_features", "atom_mask", "morgan"]
    return [torch.from_numpy(graph[field]).to(device) for field in fields]


@torch.inference_mode()
def predict(model, frame, graph_tensors, device):
    scores = np.empty(len(frame), dtype=np.float32)
    for uid, row_indices in frame.groupby("UNIPROT_ID", sort=False).indices.items():
        embedding_path = os.path.join(PROTEIN_DIR, f"{uid}.pt")
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Missing protein embedding: {embedding_path}")
        embedding = torch.load(embedding_path, map_location="cpu", weights_only=True).float()
        positions = torch.from_numpy(frame.iloc[row_indices]["Position"].to_numpy(dtype=np.int64) - 1)
        for start in range(0, len(row_indices), BATCH_SIZE):
            stop = min(start + BATCH_SIZE, len(row_indices))
            batch_indices = row_indices[start:stop]
            protein = make_windows(embedding, positions[start:stop]).to(device, non_blocking=True)
            batch_size = len(batch_indices)
            probe_inputs = [value.unsqueeze(0).expand(batch_size, *value.shape) for value in graph_tensors]
            scores[batch_indices] = model(protein, *probe_inputs).sigmoid().cpu().numpy()
        del embedding
    return scores


def calculate_metrics(labels, scores):
    positives = int(labels.sum())
    if positives == 0 or positives == len(labels):
        raise ValueError("AUROC and AP require both observed-positive and unlabeled candidates")
    prevalence = positives / len(labels)
    ap = average_precision_score(labels, scores)
    values = {
        "AUROC": roc_auc_score(labels, scores),
        "AP": ap,
        "AP_baseline": prevalence,
        "AP_fold_over_baseline": ap / prevalence,
    }
    ranking = np.argsort(-scores)
    for k in TOP_K:
        cutoff = min(k, len(labels))
        hits = int(labels[ranking[:cutoff]].sum())
        precision = hits / cutoff
        values[f"Precision@{k}"] = precision
        values[f"Recall@{k}"] = hits / positives
        values[f"EF@{k}"] = precision / prevalence
    return values


def save_seed_summary(seed_metrics):
    detail = pd.DataFrame(seed_metrics)
    detail.to_csv(os.path.join(OUTPUT_DIR, "seed_metrics.csv"), index=False)
    rows = []
    for strategy in STRATEGIES:
        group = detail[detail["Strategy"] == strategy]
        row = {"Source_dataset": SOURCE_DATASET, "Case_dataset": CASE_DATASET, "Probe": group["Probe"].iloc[0], "Target_residue": "R", "Strategy": strategy, "Models": len(group)}
        for metric in METRIC_NAMES:
            row[metric] = f"{group[metric].mean():.3f} ± {group[metric].std():.3f}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, "seed_metrics_summary.csv"), index=False, encoding="utf-8-sig")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    frame = load_case_dataset()
    probe_graphs = load_probe_graphs(FEATURE_DIR)
    probe = frame["Probe"].iloc[0]
    if probe not in probe_graphs:
        raise KeyError(f"Probe graph not found for {probe!r}; rebuild model_src/features/probe_graphs.npz")
    graph = probe_graphs[probe]
    graph_tensors = prepare_graph(graph, device)
    atom_dim = graph["atom_features"].shape[-1]
    labels = frame["Label"].to_numpy(dtype=np.int64)
    seed_metrics, ensemble_metrics = [], []
    metadata = {"Source_dataset": SOURCE_DATASET, "Case_dataset": CASE_DATASET, "Probe": probe, "Target_residue": "R"}
    print(f"Device: {device}; physical GPU: {GPU_ID}; candidates: {len(frame):,}; positives: {labels.sum():,}")

    for strategy in STRATEGIES:
        score_matrix = np.empty((len(SEEDS), len(frame)), dtype=np.float32)
        for seed_index, seed in enumerate(SEEDS):
            checkpoint = os.path.join(MODEL_DIR, strategy, f"seed_{seed}", f"model_seed_{seed}.pt")
            print(f"Scoring {strategy} | seed={seed} | {checkpoint}")
            model = ProbeSiteModel(atom_dim=atom_dim).to(device)
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
            model.eval()
            scores = predict(model, frame, graph_tensors, device)
            score_matrix[seed_index] = scores
            seed_metrics.append({**metadata, "Strategy": strategy, "Seed": seed, "N": len(frame), "Observed_positive": int(labels.sum()), "Unlabeled": int((labels == 0).sum()), **calculate_metrics(labels, scores)})
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        score_mean = score_matrix.mean(axis=0)
        score_std = score_matrix.std(axis=0, ddof=1)
        ensemble_metrics.append({**metadata, "Strategy": strategy, "Models": len(SEEDS), "N": len(frame), "Observed_positive": int(labels.sum()), "Unlabeled": int((labels == 0).sum()), **calculate_metrics(labels, score_mean)})
        predictions = frame.copy()
        predictions["Score_mean"] = score_mean
        predictions["Score_std"] = score_std
        predictions = predictions.sort_values("Score_mean", ascending=False).reset_index(drop=True)
        predictions.insert(0, "Rank", np.arange(1, len(predictions) + 1))
        predictions.to_csv(os.path.join(OUTPUT_DIR, f"{strategy}_predictions.csv.gz"), index=False, compression="gzip")
        predictions.head(max(TOP_K)).to_csv(os.path.join(OUTPUT_DIR, f"{strategy}_top_{max(TOP_K)}.csv"), index=False)
        predictions[predictions["Label"] == 1].to_csv(os.path.join(OUTPUT_DIR, f"{strategy}_observed_positive_ranks.csv"), index=False)

    save_seed_summary(seed_metrics)
    pd.DataFrame(ensemble_metrics).to_csv(os.path.join(OUTPUT_DIR, "ensemble_metrics.csv"), index=False)
    print(f"Finished. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
