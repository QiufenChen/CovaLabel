"""Evaluate trained CovaLabel models on complete external datasets."""

import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader


# ======================== modify here ========================
PROJECT_DIR = "/home/qfchen/ProbePred"
PROTEIN_FEATURE_ROOT = "/data/qfchen/esm2_feature_probe"

NEGATIVE_RATIOS = [5]
DATASETS = {"Human_Lysate_100uM": "Human", 
            "SA59_Lysate_100uM": "SA59", 
            "SA59_Lysate_1mM": "SA59", 
            "SA59_Insitu_100uM": "SA59"}
STRATEGIES = ["random", "cold_protein", "cold_probe"]
SEEDS = [42, 123, 2024, 3407, 8888]
BATCH_SIZE = 64
THRESHOLD = 0.5
# =============================================================


MODEL_ROOT = os.path.join(PROJECT_DIR, "model_src", "models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "model_src", "cross_test")
FEATURE_DIR = os.path.join(PROJECT_DIR, "model_src", "features")
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AP"]
MODEL_INPUTS = ["protein", "atom_features", "reaction_roles", "adjacency", "bond_features", "atom_mask", "morgan"]

sys.path.insert(0, os.path.join(PROJECT_DIR, "model_src"))
from data import ProbeSiteDataset, load_probe_graphs  # noqa: E402
from model import ProbeSiteModel  # noqa: E402


def load_complete_dataset(dataset, ratio, seed):
    path = os.path.join(PROJECT_DIR, "outputs", "sampled_data", dataset, ratio, f"seed_{seed}.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame.rename(columns={"Uniprot_ID": "UNIPROT_ID", "Site": "Site_type", "Postion": "Position"})


def calculate_metrics(labels, scores):
    predictions = (scores >= THRESHOLD).astype(int)
    return {
        "Accuracy": accuracy_score(labels, predictions),
        "Precision": precision_score(labels, predictions, zero_division=0),
        "Recall": recall_score(labels, predictions, zero_division=0),
        "F1": f1_score(labels, predictions, zero_division=0),
        "AUROC": roc_auc_score(labels, scores),
        "AP": average_precision_score(labels, scores),
    }


@torch.inference_mode()
def predict(model, frame, organism, probe_graphs, protein_cache, device):
    dataset = ProbeSiteDataset(frame, probe_graphs, os.path.join(PROTEIN_FEATURE_ROOT, organism))
    dataset.cache = protein_cache
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    scores = []
    for batch in loader:
        inputs = [batch[name].to(device, non_blocking=True) for name in MODEL_INPUTS]
        scores.append(model(*inputs).sigmoid().cpu().numpy().reshape(-1))
    return np.concatenate(scores)


def save_results(rows, path):
    frame = pd.DataFrame(rows)
    summary = {"Seed": "mean±std", "N": "", "Positive": "", "Unlabeled": ""}
    for metric in METRIC_NAMES:
        summary[metric] = f"{frame[metric].mean():.3f} ± {frame[metric].std():.3f}"
    pd.concat([frame, pd.DataFrame([summary])], ignore_index=True).to_csv(path, index=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    probe_graphs = load_probe_graphs(FEATURE_DIR)
    atom_dim = next(iter(probe_graphs.values()))["atom_features"].shape[-1]
    protein_caches = {"Human": {}, "SA59": {}}
    total = len(NEGATIVE_RATIOS) * len(DATASETS) * (len(DATASETS) - 1) * len(STRATEGIES)
    completed = 0
    print(f"Device: {device}; cross tests: {total}")
    for negative_ratio in NEGATIVE_RATIOS:
        ratio = f"1_{negative_ratio}"
        for source in DATASETS:
            for target, organism in DATASETS.items():
                if target == source:
                    continue
                direction_dir = os.path.join(OUTPUT_DIR, ratio, f"{source}_to_{target}")
                os.makedirs(direction_dir, exist_ok=True)
                for strategy in STRATEGIES:
                    rows = []
                    for seed in SEEDS:
                        print(source, target, strategy, seed)
                        frame = load_complete_dataset(target, ratio, seed)
                        checkpoint = os.path.join(MODEL_ROOT, ratio, source, strategy, f"seed_{seed}", f"model_seed_{seed}.pt")
                        print("Model path: ", checkpoint)
                        model = ProbeSiteModel(atom_dim=atom_dim).to(device)
                        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
                        model.eval()
                        scores = predict(model, frame, organism, probe_graphs, protein_caches[organism], device)
                        values = calculate_metrics(frame["Label"].to_numpy(), scores)
                        rows.append({"Seed": seed, "N": len(frame), "Positive": int(frame["Label"].sum()), "Unlabeled": int((frame["Label"] == 0).sum()), **values})
                        del model
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                    save_results(rows, os.path.join(direction_dir, f"{strategy}.csv"))
                    completed += 1
                    print(f"[{completed}/{total}] {ratio} | {source} -> {target} | {strategy} | saved")
    print(f"Finished. Results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
