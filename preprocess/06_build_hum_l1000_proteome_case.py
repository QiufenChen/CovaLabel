"""Build the unsampled Human_Lysate_1mM proteome-wide case dataset."""

import csv
import os

import pandas as pd


# ======================== modify here ========================
PROJECT_DIR = "/home/qfchen/ProbePred"
TARGET_RESIDUES = ["R"]
# =============================================================


DATASET_NAME = "Human_Lysate_1mM"
TRAINING_DATASET_NAME = "Human_Lysate_100uM"
FASTA_PATH = os.path.join(PROJECT_DIR, "raw_data", "HS0-2_plus_rev.fasta")
POSITIVE_PATH = os.path.join(PROJECT_DIR, "outputs", "positive_data", f"{DATASET_NAME}.csv")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs", "case_study", DATASET_NAME)
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "proteome_candidates.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "dataset_summary.csv")
LABEL_COUNT_PATH = os.path.join(OUTPUT_DIR, "label_counts.csv")
EXCLUDED_PATH = os.path.join(OUTPUT_DIR, "excluded_non_target_positives.csv")
OUTPUT_COLUMNS = ["Probe", "Uniprot_ID", "Site", "Postion", "Label"]

TRAINING_POSITIVE_PATH = os.path.join(
    PROJECT_DIR, "outputs", "positive_data", f"{TRAINING_DATASET_NAME}.csv"
)


def parse_accession(header):
    token = header.split(maxsplit=1)[0]
    if token.lower().startswith("rev_"):
        return None
    parts = token.split("|")
    return parts[1] if len(parts) >= 3 else token


def read_fasta(path):
    accession, sequence = None, []
    with open(path, encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if accession is not None:
                    yield accession, "".join(sequence).upper()
                accession, sequence = parse_accession(line[1:]), []
            elif accession is not None:
                sequence.append(line)
    if accession is not None:
        yield accession, "".join(sequence).upper()


def verify_unseen_probe(probe):
    frame = pd.read_csv(TRAINING_POSITIVE_PATH, usecols=["Probe"], encoding="utf-8-sig")
    training_probes = set(frame["Probe"].dropna().astype(str).str.strip())
    if probe in training_probes:
        raise ValueError(f"Probe {probe!r} is present in the {TRAINING_DATASET_NAME} training dataset")


def load_positives():
    positives = pd.read_csv(POSITIVE_PATH, encoding="utf-8-sig")
    required = {"Probe", "UNIPROT_ID", "Position", "Site_type"}
    if missing := required - set(positives.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")
    probes = positives["Probe"].dropna().astype(str).str.strip().unique()
    if len(probes) != 1:
        raise ValueError(f"{DATASET_NAME} must contain one probe, found {probes.tolist()}")
    probe = probes[0]
    verify_unseen_probe(probe)
    positives = positives.copy()
    positives["UNIPROT_ID"] = positives["UNIPROT_ID"].astype(str).str.strip()
    positives["Position"] = positives["Position"].astype(int)
    positives["Site_type"] = positives["Site_type"].astype(str).str.strip()
    target = positives[positives["Site_type"].isin(TARGET_RESIDUES)].copy()
    excluded = positives[~positives["Site_type"].isin(TARGET_RESIDUES)].copy()
    if target.duplicated(["UNIPROT_ID", "Position", "Site_type"]).any():
        raise ValueError("Duplicate target positive sites were found")
    return probe, positives, target, excluded


def main():
    invalid = set(TARGET_RESIDUES) - set("ACDEFGHIKLMNPQRSTVWY")
    if invalid:
        raise ValueError(f"Noncanonical target residues: {sorted(invalid)}")
    probe, all_positives, target_positives, excluded_positives = load_positives()
    positive_keys = set(target_positives[["UNIPROT_ID", "Position", "Site_type"]].itertuples(index=False, name=None))
    if not positive_keys:
        raise ValueError("No R-positive sites were found in the HUM-L1000 positive dataset")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    excluded_positives.to_csv(EXCLUDED_PATH, index=False, encoding="utf-8-sig")
    temporary_path = OUTPUT_PATH + ".tmp"
    seen_accessions, recovered = set(), set()
    protein_count = proteins_with_candidates = positive_count = unlabeled_count = 0
    with open(temporary_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for accession, sequence in read_fasta(FASTA_PATH):
            if accession in seen_accessions:
                raise ValueError(f"Duplicate forward FASTA accession: {accession}")
            seen_accessions.add(accession)
            protein_count += 1
            has_candidate = False
            for position, residue in enumerate(sequence, start=1):
                if residue not in TARGET_RESIDUES:
                    continue
                key = accession, position, residue
                label = int(key in positive_keys)
                writer.writerow({"Probe": probe, "Uniprot_ID": accession, "Site": residue, "Postion": position, "Label": label})
                has_candidate = True
                if label == 1:
                    recovered.add(key)
                    positive_count += 1
                else:
                    unlabeled_count += 1
            proteins_with_candidates += int(has_candidate)
    missing_positives = positive_keys - recovered
    if missing_positives:
        os.remove(temporary_path)
        raise ValueError(f"Positive sites missing or inconsistent with FASTA: {sorted(missing_positives)[:10]}")
    if positive_count != len(positive_keys) or unlabeled_count == 0:
        os.remove(temporary_path)
        raise ValueError(f"Invalid label counts: Label 1={positive_count}, Label 0={unlabeled_count}")
    os.replace(temporary_path, OUTPUT_PATH)
    pd.DataFrame([{"Label": 1, "Count": positive_count}, {"Label": 0, "Count": unlabeled_count}]).to_csv(LABEL_COUNT_PATH, index=False)
    terminal_mask = excluded_positives["Site_type"].str.lower().str.contains("terminal", na=False)
    summary = pd.DataFrame([{"Dataset": DATASET_NAME, "Training_dataset": TRAINING_DATASET_NAME, "Probe": probe, "Probe_status": "unseen in training dataset", "Target_residues": ";".join(TARGET_RESIDUES), "Input_positive_rows": len(all_positives), "Retained_target_positive_sites": positive_count, "Excluded_internal_non_target_positive_rows": int((~terminal_mask).sum()), "Excluded_terminal_positive_rows": int(terminal_mask.sum()), "Forward_FASTA_proteins": protein_count, "Proteins_with_candidate_sites": proteins_with_candidates, "Unlabeled_candidate_sites": unlabeled_count, "Total_candidate_sites": positive_count + unlabeled_count, "Negative_sampling": "none", "Dataset_partitioning": "none"}])
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
