"""Train CovaLabel on the fixed train/val/test CSV files."""

import copy
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.nn import functional as F
from torch.utils.data import DataLoader

from data import ProbeSiteDataset, load_probe_graphs
from model import ProbeSiteModel


# ======================== modify here ========================
PROJECT_DIR = "/home/qfchen/ProbePred"
PROTEIN_FEATURE_ROOT = "/data/qfchen/esm2_feature_probe"

NEGATIVE_RATIO = 5  # Change to 3, 5, 7, or 10.
DATASETS = ["Human_Lysate_100uM", "SA59_Lysate_100uM", "SA59_Lysate_1mM", "SA59_Insitu_100uM"]
STRATEGIES = ["random", "cold_protein", "cold_probe"]
SEEDS = [42, 123, 2024, 3407, 8888]

EPOCHS = 100
PATIENCE = 15
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
UNLABELED_WEIGHT = 0.3
THRESHOLD = 0.5
# =============================================================


RATIO = f"1_{NEGATIVE_RATIO}"
SPLIT_ROOT = os.path.join(PROJECT_DIR, "outputs", "split_data")
MODEL_ROOT = os.path.join(PROJECT_DIR, "model_src", "models", RATIO)
FEATURE_DIR = os.path.join(PROJECT_DIR, "model_src", "outputs", "features")
INPUT_NAMES = ["protein", "atom_features", "reaction_roles", "adjacency", "bond_features", "atom_mask", "morgan"]
METRIC_NAMES = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AP"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(dataset, strategy, seed, split):
    path = os.path.join(SPLIT_ROOT, dataset, RATIO, strategy, f"seed_{seed}", f"{split}.csv")
    frame = pd.read_csv(path, encoding="utf-8-sig")
    return frame.rename(columns={"Uniprot_ID": "UNIPROT_ID", "Site": "Site_type", "Postion": "Position"})


def make_loader(frame, probe_graphs, protein_dir, shuffle, device):
    dataset = ProbeSiteDataset(frame, probe_graphs, protein_dir)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0, pin_memory=device.type == "cuda")


def weighted_loss(logits, labels):
    weights = torch.where(labels == 1, 1.0, UNLABELED_WEIGHT)
    losses = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (losses * weights).sum() / weights.sum()


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
def evaluate(model, loader, device):
    model.eval()
    labels, scores, indices = [], [], []
    for batch in loader:
        logits = model(*[batch[name].to(device, non_blocking=True) for name in INPUT_NAMES])
        labels.append(batch["label"].numpy())
        scores.append(logits.sigmoid().cpu().numpy().reshape(-1))
        indices.append(batch["index"].numpy())
    labels, scores, indices = np.concatenate(labels), np.concatenate(scores), np.concatenate(indices)
    return calculate_metrics(labels, scores), scores, indices


def train_model(seed, train_frame, val_loader, probe_graphs, protein_dir, atom_dim, device):
    set_seed(seed)
    model = ProbeSiteModel(atom_dim=atom_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_loader = make_loader(train_frame, probe_graphs, protein_dir, True, device)
    best_ap, best_state, wait, history = -1.0, None, 0, []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*[batch[name].to(device, non_blocking=True) for name in INPUT_NAMES])
            loss = weighted_loss(logits, batch["label"].to(device, non_blocking=True))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch["label"])
        val_metrics, _, _ = evaluate(model, val_loader, device)
        history.append({"Epoch": epoch, "Train_loss": total_loss / len(train_frame), "Val_AUROC": val_metrics["AUROC"], "Val_AP": val_metrics["AP"]})
        print(f"seed={seed} epoch={epoch:03d} loss={history[-1]['Train_loss']:.4f} val_AP={val_metrics['AP']:.4f}")
        if val_metrics["AP"] > best_ap:
            best_ap, best_state, wait = val_metrics["AP"], copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_ap


def save_summary(results):
    detail = pd.DataFrame(results)
    detail.to_csv(os.path.join(MODEL_ROOT, "training_results.csv"), index=False)
    summary = detail.groupby(["Dataset", "Strategy"])[METRIC_NAMES].agg(["mean", "std"]).reset_index()
    summary.columns = [column if isinstance(column, str) else "_".join(part for part in column if part) for column in summary.columns]
    for metric in METRIC_NAMES:
        summary[f"{metric}_mean_std"] = summary.apply(lambda row: f"{row[f'{metric}_mean']:.3f} ± {row[f'{metric}_std']:.3f}", axis=1)
    summary.to_csv(os.path.join(MODEL_ROOT, "training_summary.csv"), index=False)


def main():
    os.makedirs(MODEL_ROOT, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_graphs = load_probe_graphs(FEATURE_DIR)
    atom_dim = next(iter(probe_graphs.values()))["atom_features"].shape[-1]
    results = []
    print(f"Device: {device}; ratio: {RATIO}")
    for dataset in DATASETS:
        organism = "Human" if dataset.startswith("Human_") else "SA59"
        protein_dir = os.path.join(PROTEIN_FEATURE_ROOT, organism)
        for strategy in STRATEGIES:
            for seed in SEEDS:
                print(f"Training {dataset} | {strategy} | seed={seed}")
                train_frame = load_split(dataset, strategy, seed, "train")
                val_frame = load_split(dataset, strategy, seed, "val")
                test_frame = load_split(dataset, strategy, seed, "test")
                val_loader = make_loader(val_frame, probe_graphs, protein_dir, False, device)
                test_loader = make_loader(test_frame, probe_graphs, protein_dir, False, device)
                model, history, best_val_ap = train_model(seed, train_frame, val_loader, probe_graphs, protein_dir, atom_dim, device)
                test_metrics, scores, indices = evaluate(model, test_loader, device)
                result_dir = os.path.join(MODEL_ROOT, dataset, strategy, f"seed_{seed}")
                os.makedirs(result_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(result_dir, f"model_seed_{seed}.pt"))
                history.to_csv(os.path.join(result_dir, "history.csv"), index=False)
                predictions = test_frame.iloc[indices].copy()
                predictions["Score"] = scores
                predictions.to_csv(os.path.join(result_dir, "test_predictions.csv"), index=False)
                row = {"Ratio": RATIO, "Dataset": dataset, "Strategy": strategy, "Seed": seed, "Best_val_AP": best_val_ap, **test_metrics}
                pd.DataFrame([row]).to_csv(os.path.join(result_dir, "metrics.csv"), index=False)
                results.append(row)
                pd.DataFrame(results).to_csv(os.path.join(MODEL_ROOT, "training_results.csv"), index=False)
                del model, val_loader, test_loader
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    save_summary(results)
    print(f"Finished. Models saved to: {MODEL_ROOT}")


if __name__ == "__main__":
    main()
