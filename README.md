# CovaLabel

CovaLabel is a lightweight deep-learning framework for predicting whether a protein residue can be labeled by a covalent chemical probe. It combines local protein context from ESM-2 with a reaction-aware molecular graph representation of the probe.

The repository supports standard random evaluation, cold-protein evaluation, cold-probe evaluation, multiple positive-to-unlabeled ratios, multi-seed training, and cross-dataset transfer evaluation.

> **Research status:** CovaLabel is research code. Samples assigned label `0` are unlabeled candidates, not experimentally confirmed negatives. Model scores should therefore not be interpreted as calibrated probabilities of covalent labeling.

## Model overview

CovaLabel uses two input branches:

### Protein-site branch

- residue-level embeddings from `esm2_t33_650M_UR50D`;
- a 21-residue window centered on the candidate site;
- learned relative-position embeddings;
- multi-scale 1D convolutions with kernel sizes 3, 5, and 7;
- center-residue and attention-weighted window pooling.

### Chemical-probe branch

- a molecular graph constructed with RDKit;
- 24-dimensional atom features;
- six bond features: single, double, triple, aromatic, conjugated, and ring bonds;
- SMARTS-derived atom-level reaction-role annotations;
- two bond-aware graph message-passing blocks;
- a 2,048-bit radius-2 Morgan fingerprint;
- whole-molecule and reactive-atom pooling.

The protein representation attends to probe atoms through multi-head cross-attention. The final classifier uses the site representation, probe-conditioned site representation, probe representation, element-wise interaction, and absolute difference.

## Repository contents

```text
CovaLabel/
├── main_model/                # Model source and feature archive
│   ├── get_feature.py              # Build probe graphs and Morgan fingerprints
│   ├── build_protein_embeddings.py # Build residue-level ESM-2 embeddings
│   ├── data.py                     # Load splits and assemble model inputs
│   ├── model.py                    # CovaLabel architecture
│   ├── train.py                    # Multi-dataset, multi-seed training
│   └── cross_dataset_eval.py       # Cross-dataset transfer evaluation
├── preprocess/                # Data preparation notebooks and case builder
├── datasets/                  # Raw data, sequences, sampled data, and splits
├── case_study/                # Evaluation, plotting scripts, and results
└── README.md
```

The repository includes probe features in `main_model/features` and case-study results in `case_study/results`. Protein embeddings and trained checkpoints must be generated locally or supplied separately.

## Requirements

- Python 3.10 or later
- PyTorch
- NumPy
- pandas
- scikit-learn
- RDKit
- openpyxl
- tqdm
- fair-esm, only when generating ESM-2 embeddings

Example installation:

```bash
conda create -n covalabel python=3.10 -y
conda activate covalabel

conda install -c conda-forge rdkit openpyxl -y
pip install torch numpy pandas scikit-learn tqdm fair-esm
```

CovaLabel does not require PyTorch Geometric, ChemBERTa, Transformers, or Mamba.

## Expected data

### Probe information

`ProbeInfo.xlsx` must contain the following columns:

| Column | Description |
|---|---|
| `Probe` | Unique probe name |
| `SMILES` | Probe SMILES string |

Default location in the original workspace:

```text
datasets/raw_data/ProbeInfo.xlsx
```

### Split tables

Training expects pre-generated CSV files with this layout:

```text
datasets/split_data/
└── <dataset>/
    └── 1_<negative_ratio>/
        └── <strategy>/
            └── seed_<seed>/
                ├── train.csv
                ├── val.csv
                └── test.csv
```

Required CSV columns:

| Column | Description |
|---|---|
| `Probe` | Probe name matching `ProbeInfo.xlsx` |
| `Uniprot_ID` | UniProt accession |
| `Site` | Candidate residue type |
| `Postion` | One-based residue position; the spelling is retained for compatibility with the current data files |
| `Label` | `1` for observed positive and `0` for an unlabeled candidate |

The current training script includes these datasets:

```text
Human_Lysate_100uM
SA59_Lysate_100uM
SA59_Lysate_1mM
SA59_Insitu_100uM
```

### ESM-2 embeddings

Each protein must have one tensor file:

```text
<protein_feature_root>/
├── Human/<UniProt_ID>.pt
└── SA59/<UniProt_ID>.pt
```

Every tensor must have shape:

```text
[protein_length, 1280]
```

The existing configuration uses:

```text
/data/qfchen/esm2_feature_probe/Human/
/data/qfchen/esm2_feature_probe/SA59/
```

If embeddings already exist, `build_protein_embeddings.py` can be skipped.

## Path configuration

The scripts retain absolute paths from the original research workspace. Configure them before running. All commands below assume the repository root as the working directory.

Set `PROJECT_DIR` to the absolute path of your local CovaLabel checkout and `PROTEIN_FEATURE_ROOT` to the directory containing `Human/` and `SA59/` embedding folders. Update derived paths in each script as follows:

| Resource | Repository-relative path |
|---|---|
| Probe metadata | `datasets/raw_data/ProbeInfo.xlsx` |
| Split root | `datasets/split_data` |
| Sampled datasets | `datasets/sampled_data` |
| Probe features | `main_model/features` |
| Checkpoints | `main_model/models/<ratio>` |
| Transfer results | `main_model/cross_test` |
| Model import directory | `main_model` |
| Case candidates | `datasets/case_study/Human_Lysate_1mM` |
| Case results | `case_study/results/<ratio>` |

Replace legacy `model_src` references with `main_model` and legacy `outputs` data references with `datasets`. Explicitly set the feature output in `get_feature.py` and `FEATURE_DIR` in training and evaluation to `main_model/features`: the original scripts use inconsistent feature paths.

For `build_protein_embeddings.py`, configure `INPUT_PATH`, `OUTPUT_DIR`, and `TORCH_HOME`. Sequence workbooks are `datasets/fasta/Huamn.xlsx` (original spelling) and `datasets/fasta/SA59.xlsx`, with `UNIPROTID` and `SEQUENCE` columns. Run separately for each organism, changing the output folder accordingly.

## Data preparation

Prepared datasets and split tables are included under `datasets/`. To rebuild them, adapt the notebook paths and run `preprocess/01_get_pos_data.ipynb` through `05_split_dataset.ipynb` in numeric order. These extract positives, parse sequences, build unlabeled candidates, sample candidates, and create splits.

Included sampling and split summaries cover ratios 1:3, 1:5, 1:7, and 1:10. `Human_Lysate_1mM` is included for the case study but is not in the default four-dataset training list.

## Build probe features

After configuring `PROJECT_DIR`, run:

```bash
python main_model/get_feature.py
```

This creates `probe_graphs.npz`, containing:

- padded atom features;
- atom-level reaction-role masks;
- molecular adjacency matrices;
- bond features;
- atom masks;
- Morgan fingerprints;
- probe names and canonical SMILES.

Regenerate this file whenever `ProbeInfo.xlsx`, atom features, SMARTS definitions, or Morgan settings change.

## Build ESM-2 embeddings

Skip this step if compatible residue-level ESM-2 tensors already exist.

Edit the input and output paths in `build_protein_embeddings.py`, then run:

```bash
python main_model/build_protein_embeddings.py
```

The script uses `esm2_t33_650M_UR50D`, extracts layer 33, stores one `float16` tensor per protein, and skips proteins whose output files already exist.

## Train CovaLabel

Configure the experiment near the top of `train.py`:

```python
NEGATIVE_RATIO = 5
DATASETS = [
    "Human_Lysate_100uM",
    "SA59_Lysate_100uM",
    "SA59_Lysate_1mM",
    "SA59_Insitu_100uM",
]
STRATEGIES = ["random", "cold_protein", "cold_probe"]
SEEDS = [42, 123, 2024, 3407, 8888]
```

Then run:

```bash
python main_model/train.py
```

For a quick smoke test, start with one dataset, one strategy, one seed, and two epochs:

```python
DATASETS = ["SA59_Lysate_100uM"]
STRATEGIES = ["random"]
SEEDS = [42]
EPOCHS = 2
```

### Split strategies

| Strategy | Evaluation target |
|---|---|
| `random` | Randomly held-out probe-site samples |
| `cold_protein` | Generalization to proteins excluded from training |
| `cold_probe` | Generalization to probes excluded from training |

The split CSV files are treated as fixed inputs. CovaLabel does not recreate the split during model training.

### Training objective

CovaLabel uses weighted binary cross-entropy:

- observed positives receive weight `1.0`;
- unlabeled candidates receive `UNLABELED_WEIGHT`, currently `0.3`.

The best checkpoint is selected by validation average precision. Early stopping uses the configured patience, and gradients are clipped to a maximum norm of `1.0`.

### Training outputs

Outputs are organized by negative ratio, dataset, split strategy, and seed:

```text
models/
└── 1_<negative_ratio>/
    └── <dataset>/
        └── <strategy>/
            └── seed_<seed>/
                ├── model_seed_<seed>.pt
                ├── history.csv
                ├── metrics.csv
                └── test_predictions.csv
```

Aggregate files include:

```text
training_results.csv
training_summary.csv
```

Reported metrics are accuracy, precision, recall, F1, AUROC, and average precision. Threshold-dependent metrics use a default threshold of `0.5`.

## Cross-dataset evaluation

`cross_dataset_eval.py` evaluates a model trained on one dataset against a complete external target dataset.

Configure:

```python
NEGATIVE_RATIOS = [5]
DATASETS = {
    "Human_Lysate_100uM": "Human",
    "SA59_Lysate_100uM": "SA59",
    "SA59_Lysate_1mM": "SA59",
    "SA59_Insitu_100uM": "SA59",
}
```

The script expects target datasets at:

```text
datasets/sampled_data/<dataset>/1_<negative_ratio>/seed_<seed>.csv
```

Run:

```bash
python main_model/cross_dataset_eval.py
```

Results are stored by transfer direction:

```text
cross_test/
└── 1_<negative_ratio>/
    └── <source>_to_<target>/
        ├── random.csv
        ├── cold_protein.csv
        └── cold_probe.csv
```

## Reproducibility

The default experiment uses five seeds:

```text
42, 123, 2024, 3407, 8888
```

Python, NumPy, PyTorch CPU, and PyTorch CUDA random generators are seeded. For strict GPU determinism, additional PyTorch deterministic settings may be required and can reduce performance.

## Important limitations

1. Label `0` represents an unlabeled candidate, not a confirmed non-reactive site.
2. The current weighted objective is a positive–unlabeled proxy, not an unbiased PU-risk estimator.
3. SMARTS-derived reaction roles are computational annotations and may not identify the experimentally active atom in every probe.
4. Accuracy, precision, recall, and F1 depend on the selected threshold and class ratio. Average precision should be interpreted together with the positive fraction.
5. Cross-dataset performance can reflect changes in organism, sample preparation, probe coverage, concentration, and candidate sampling.

## Proteome-wide case study

The case study evaluates models trained on `Human_Lysate_100uM` against arginine (`R`) candidates from `Human_Lysate_1mM`.

After configuring paths in each script, run:

```bash
# Optional: rebuild the included candidate table.
python preprocess/06_build_hum_l1000_proteome_case.py

# Requires checkpoints and protein embeddings for the configured experiments.
python case_study/run_hum_l1000_case.py
```

In the candidate builder, point FASTA inputs to `datasets/raw_data`, positive tables to `datasets/positive_data`, and outputs to `datasets/case_study/Human_Lysate_1mM`. In the evaluator, update the model, feature, candidate, output, and import paths, as well as `GPU_ID`.

Outputs include per-seed and ensemble metrics, observed-positive ranks, and top-ranked candidates. Plotting scripts are `case_study/plot_hum_l1000_panels_cd.py` and `case_study/plot_hum_l1000_panel_e.py`; configure their paths and install their plotting dependencies before use.

## Citation

Publication details for CovaLabel have not yet been added to this repository.

## License

No license file is currently included. Dependencies, source datasets, and pretrained ESM-2 weights are subject to their respective licenses and terms.