"""Build padded probe graphs, atom-level reaction roles and Morgan fingerprints."""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


PROJECT_DIR = "/home/qfchen/ProbePred"
MORGAN_BITS = 2048
ELEMENTS = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "Si", "Se"]
REACTION_SMARTS = {
    "haloacetamide": "[F,Cl,Br,I]CC(=O)N",
    "alkyl_halide": "[C;X4]-[F,Cl,Br,I]",
    "aryl_halide": "c-[F,Cl,Br,I]",
    "michael_acceptor": "[C,c]=[C,c]-[C,S](=O)-[#6,#7,#8]",
    "sulfonyl_fluoride": "S(=O)(=O)F",
    "epoxide_aziridine": "[O,N;r3]1[C;r3][C;r3]1",
    "isocyanate_isothiocyanate": "N=C=[O,S]",
    "aldehyde": "[C;H1](=O)[#6]",
    "activated_ester": "C(=O)O[N,S]",
    "nitrile": "C#N",
    "terminal_alkyne": "C#[C;H1]",
    "tetrazine_azo": "[n,N]1[n,N][c,C][n,N][n,N][c,C]1",
    "annotation_available": "*",
}


def atom_features(atom):
    element = [float(atom.GetSymbol() == value) for value in ELEMENTS]
    element.append(float(atom.GetSymbol() not in ELEMENTS))
    degree = [float(atom.GetDegree() == value) for value in range(5)]
    hybridization = [
        float(atom.GetHybridization() == value)
        for value in [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3]
    ]
    return element + degree + [
        atom.GetFormalCharge() / 3,
        float(atom.GetIsAromatic()),
        float(atom.IsInRing()),
    ] + hybridization


def reaction_roles(molecule, queries):
    roles = np.zeros((molecule.GetNumAtoms(), len(queries)), dtype=np.float32)
    for role, query in enumerate(queries):
        for match in molecule.GetSubstructMatches(query):
            roles[list(match), role] = 1
    return roles


def bond_features(bond):
    bond_type = bond.GetBondType()
    return [
        float(bond_type == Chem.BondType.SINGLE),
        float(bond_type == Chem.BondType.DOUBLE),
        float(bond_type == Chem.BondType.TRIPLE),
        float(bond_type == Chem.BondType.AROMATIC),
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


def main():
    probe_file = os.path.join(PROJECT_DIR, "raw_data", "ProbeInfo.xlsx")
    output_dir = os.path.join(PROJECT_DIR, "model_src", "outputs", "features")
    os.makedirs(output_dir, exist_ok=True)
    probes = pd.read_excel(probe_file, usecols=["Probe", "SMILES"]).dropna().drop_duplicates("Probe")
    molecules = [Chem.MolFromSmiles(value.strip()) for value in probes.SMILES]
    if any(value is None for value in molecules):
        raise ValueError("ProbeInfo.xlsx contains an invalid SMILES")

    max_atoms = max(value.GetNumAtoms() for value in molecules)
    atom_dim = len(atom_features(molecules[0].GetAtomWithIdx(0)))
    atom_array = np.zeros((len(probes), max_atoms, atom_dim), dtype=np.float32)
    role_array = np.zeros((len(probes), max_atoms, len(REACTION_SMARTS)), dtype=np.float32)
    adjacency = np.zeros((len(probes), max_atoms, max_atoms), dtype=np.float32)
    bond_array = np.zeros((len(probes), max_atoms, max_atoms, 6), dtype=np.float32)
    atom_mask = np.zeros((len(probes), max_atoms), dtype=bool)
    morgan = np.zeros((len(probes), MORGAN_BITS), dtype=np.float32)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=MORGAN_BITS)
    queries = [Chem.MolFromSmarts(value) for value in REACTION_SMARTS.values()]

    for index, molecule in enumerate(molecules):
        count = molecule.GetNumAtoms()
        atom_array[index, :count] = np.asarray([atom_features(atom) for atom in molecule.GetAtoms()])
        role_array[index, :count] = reaction_roles(molecule, queries)
        atom_mask[index, :count] = True
        for bond in molecule.GetBonds():
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            adjacency[index, left, right] = adjacency[index, right, left] = 1
            features = bond_features(bond)
            bond_array[index, left, right] = bond_array[index, right, left] = features
        adjacency[index, np.arange(count), np.arange(count)] = 1
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(molecule), morgan[index])

    np.savez_compressed(
        os.path.join(output_dir, "probe_graphs.npz"),
        probe_names=np.asarray(probes.Probe.astype(str).tolist(), dtype=str),
        smiles=np.asarray([Chem.MolToSmiles(value) for value in molecules], dtype=str),
        atom_features=atom_array,
        reaction_roles=role_array,
        adjacency=adjacency,
        bond_features=bond_array,
        atom_mask=atom_mask,
        morgan=morgan,
        reaction_names=np.asarray(list(REACTION_SMARTS), dtype=str),
    )
    print(f"Saved {len(probes)} probe graphs; max_atoms={max_atoms}; atom_dim={atom_dim}")


if __name__ == "__main__":
    main()
