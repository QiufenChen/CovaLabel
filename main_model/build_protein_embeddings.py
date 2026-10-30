"""Build residue-level ESM-2 embeddings for the complete human proteome."""

import os

import esm
import pandas as pd
import torch
from tqdm import tqdm


# ======================== modify here ========================
INPUT_PATH = "/home/qfchen/ProbePred/outputs/fasta/Huamn.xlsx"
OUTPUT_DIR = "/data/qfchen/esm2_feature_probe/Human"
TORCH_HOME = "/home/qfchen/lager_model"
CHUNK_SIZE = 900
# =============================================================


REPRESENTATION_LAYER = 33


def embed(sequence, model, converter, device):
    pieces = []
    for start in range(0, len(sequence), CHUNK_SIZE):
        part = sequence[start:start + CHUNK_SIZE]
        _, _, tokens = converter([("protein", part)])
        with torch.inference_mode():
            output = model(tokens.to(device), repr_layers=[REPRESENTATION_LAYER], return_contacts=False)
        pieces.append(output["representations"][REPRESENTATION_LAYER][0, 1:len(part) + 1].float().cpu())
    return torch.cat(pieces).half()


def main():
    os.environ["TORCH_HOME"] = TORCH_HOME
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.eval().to(device)
    converter = alphabet.get_batch_converter()
    table = pd.read_excel(INPUT_PATH, usecols=["UNIPROTID", "SEQUENCE"]).dropna()
    table["UNIPROTID"] = table["UNIPROTID"].astype(str).str.strip()
    table["SEQUENCE"] = table["SEQUENCE"].astype(str).str.strip().str.upper()
    if table.groupby("UNIPROTID")["SEQUENCE"].nunique().gt(1).any():
        raise ValueError("The same UNIPROTID is associated with different sequences")
    table = table.drop_duplicates("UNIPROTID").reset_index(drop=True)
    existing_ids = {os.path.splitext(name)[0] for name in os.listdir(OUTPUT_DIR) if name.endswith(".pt")}
    missing_table = table[~table["UNIPROTID"].isin(existing_ids)].reset_index(drop=True)
    print(f"Model: ESM-2 t33 650M; representation layer: {REPRESENTATION_LAYER}; device: {device}")
    print(f"Input proteins: {len(table)}; existing embeddings: {len(table) - len(missing_table)}; embeddings to build: {len(missing_table)}")
    print(f"Output: {OUTPUT_DIR}")

    for row in tqdm(missing_table.itertuples(index=False), total=len(missing_table), desc="Human"):
        accession, sequence = row.UNIPROTID, row.SEQUENCE
        if not sequence:
            raise ValueError(f"Empty sequence: {accession}")
        output_path = os.path.join(OUTPUT_DIR, f"{accession}.pt")
        temporary_path = output_path + ".tmp"
        torch.save(embed(sequence, model, converter, device), temporary_path)
        os.replace(temporary_path, output_path)

    print(f"Finished. New embeddings: {len(missing_table)}; saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
