"""Load fixed protein-disjoint datasets and model features."""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_split(
    project_dir, organism, treatment, concentration, split,
    split_strategy="random", negative_ratio=5, sampling_seed=42,
):
    dataset_name = f"{organism}_{treatment}_{concentration}"
    folder = os.path.join(
        project_dir, "outputs", "split_data", dataset_name, f"1_{negative_ratio}",
        split_strategy, f"seed_{sampling_seed}",
    )
    frame = pd.read_csv(os.path.join(folder, f"{split}.csv"))
    return frame.rename(columns={
        "Uniprot_ID": "UNIPROT_ID",
        "Site": "Site_type",
        "Postion": "Position",
    })


def load_probe_graphs(feature_dir):
    values = np.load(os.path.join(feature_dir, "probe_graphs.npz"), allow_pickle=False)
    names = values["probe_names"].tolist()
    fields = [
        "atom_features", "reaction_roles", "adjacency",
        "bond_features", "atom_mask", "morgan",
    ]
    return {name: {field: values[field][index] for field in fields} for index, name in enumerate(names)}


def site_window(value, position, size=21):
    value = torch.as_tensor(value, dtype=torch.float32)
    center, half = int(position) - 1, size // 2
    start, end = center - half, center + half + 1
    left, right = max(0, start), min(len(value), end)
    output = torch.zeros(size, value.shape[1])
    output[left - start:left - start + right - left] = value[left:right]
    return output


class ProbeSiteDataset(Dataset):
    def __init__(self, frame, probe_graphs, protein_dir):
        self.data = frame.reset_index(drop=True)
        self.probe_graphs = probe_graphs
        self.protein_dir = protein_dir
        self.cache = {}

    def __len__(self):
        return len(self.data)

    def protein(self, uid):
        if uid not in self.cache:
            self.cache[uid] = torch.load(
                os.path.join(self.protein_dir, f"{uid}.pt"), map_location="cpu", weights_only=True
            ).float()
        return self.cache[uid]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        graph = self.probe_graphs[row.Probe]
        return {
            "protein": site_window(self.protein(row.UNIPROT_ID), row.Position),
            "atom_features": torch.from_numpy(graph["atom_features"]).float(),
            "reaction_roles": torch.from_numpy(graph["reaction_roles"]).float(),
            "adjacency": torch.from_numpy(graph["adjacency"]).float(),
            "bond_features": torch.from_numpy(graph["bond_features"]).float(),
            "atom_mask": torch.from_numpy(graph["atom_mask"]),
            "morgan": torch.from_numpy(graph["morgan"]).float(),
            "label": torch.tensor(float(row.Label)),
            "index": index,
        }
