"""Lightweight ESM + molecular graph model for probe-site prediction."""

import torch
from torch import nn
from torch.nn import functional as F


class ProteinEncoder(nn.Module):
    def __init__(self, input_dim=1280, hidden=128):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, hidden, kernel, padding=kernel // 2) for kernel in (3, 5, 7)
        ])
        self.output = nn.Linear(hidden * 3, hidden)
        self.position = nn.Embedding(21, hidden)

    def forward(self, x, mask):
        x = self.norm(x).transpose(1, 2)
        x = torch.cat([F.gelu(conv(x)) for conv in self.convs], dim=1).transpose(1, 2)
        x = self.output(x)
        positions = torch.arange(x.size(1), device=x.device)
        return (x + self.position(positions)[None]) * mask.unsqueeze(-1)


class GraphBlock(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.node_message = nn.Linear(hidden, hidden)
        self.bond_message = nn.Linear(6, hidden, bias=False)
        self.update = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Dropout(0.1))
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, adjacency, bonds, mask):
        degree = adjacency.sum(-1, keepdim=True).clamp_min(1)
        node = self.node_message(x).unsqueeze(1)
        edge = self.bond_message(bonds)
        message = ((node + edge) * adjacency.unsqueeze(-1)).sum(2) / degree
        return self.norm(x + self.update(torch.cat([x, message], dim=-1))) * mask.unsqueeze(-1)


class ProbeGraphEncoder(nn.Module):
    def __init__(self, atom_dim=24, reaction_dim=13, hidden=128):
        super().__init__()
        self.atom = nn.Linear(atom_dim, hidden)
        self.reaction = nn.Linear(reaction_dim, hidden)
        self.blocks = nn.ModuleList([GraphBlock(hidden), GraphBlock(hidden)])
        self.morgan = nn.Sequential(nn.Linear(2048, 256), nn.GELU(), nn.Dropout(0.2), nn.Linear(256, hidden))
        self.fusion = nn.Linear(hidden * 3, hidden)

    def forward(self, atoms, roles, adjacency, bonds, mask, morgan):
        x = self.atom(atoms) + self.reaction(roles)
        for block in self.blocks:
            x = block(x, adjacency, bonds, mask)
        graph = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        reactive_mask = (roles[..., :-1].sum(-1) > 0) & mask
        fallback_mask = torch.where(reactive_mask.any(1, keepdim=True), reactive_mask, mask)
        reactive = (x * fallback_mask.unsqueeze(-1)).sum(1) / fallback_mask.sum(1, keepdim=True)
        graph = F.gelu(self.fusion(torch.cat([graph, reactive, self.morgan(morgan)], dim=-1)))
        return x, graph


class ProbeSiteModel(nn.Module):
    def __init__(self, protein_dim=1280, atom_dim=24, hidden=128, window_size=21):
        super().__init__()
        self.protein = ProteinEncoder(protein_dim, hidden)
        self.probe = ProbeGraphEncoder(atom_dim, 13, hidden)
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden)
        self.gate = nn.Linear(hidden, hidden)
        self.center = window_size // 2
        self.site_score = nn.Linear(hidden, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 5, hidden), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1),
        )

    def forward(self, protein, atoms, roles, adjacency, bonds, atom_mask, morgan):
        protein_mask = protein.abs().sum(-1) > 0
        site_tokens = self.protein(protein, protein_mask)
        atom_tokens, probe = self.probe(atoms, roles, adjacency, bonds, atom_mask, morgan)
        interaction, _ = self.attention(
            site_tokens, atom_tokens, atom_tokens, key_padding_mask=~atom_mask
        )
        site_tokens = self.attention_norm(site_tokens + interaction)
        scores = self.site_score(site_tokens).squeeze(-1).masked_fill(~protein_mask, -1e4)
        site = 0.5 * site_tokens[:, self.center] + 0.5 * (
            site_tokens * scores.softmax(-1).unsqueeze(-1)
        ).sum(1)
        conditioned_site = site * torch.sigmoid(self.gate(probe))
        features = torch.cat([
            site, conditioned_site, probe, site * probe, (site - probe).abs()
        ], dim=-1)
        return self.classifier(features).squeeze(-1)
