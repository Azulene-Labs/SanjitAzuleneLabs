import pytest
import pandas as pd
import numpy as np
from rdkit import Chem
    


# 1) Define patterns for the unnatural residues you care about
#    (using SMARTS or SMILES as patterns; SMARTS is more flexible)

inclusiveDB = pd.read_csv("uaa_data/inclusive_db.csv", sep=";")

UNNATURAL_RESIDUES = inclusiveDB[['ncAA abbreviation(s) used in the publication', 'ncAA SMILES notation']]
UNNATURAL_RESIDUES = UNNATURAL_RESIDUES[~UNNATURAL_RESIDUES.apply(lambda row: row.astype(str).str.contains("not available", case=False)).any(axis=1)]
UNNATURAL_RESIDUES = dict(zip(UNNATURAL_RESIDUES.iloc[:, 0], UNNATURAL_RESIDUES.iloc[:, 1]))


def find_unnatural_residues_in_smiles(smiles: str,
                                      patterns: dict[str, str] = UNNATURAL_RESIDUES):
    """
    Return a list of unnatural amino acids whose substructure
    appears in the given peptide SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles!r}")

    found = []
    for name, smarts in patterns.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            raise ValueError(f"Invalid SMARTS/SMILES pattern for {name}: {smarts!r}")

        if mol.HasSubstructMatch(patt):
            found.append(name)

    return found

@pytest.fixture
def contains_unnatural_residue(smiles: str,
                               patterns: dict[str, str] = UNNATURAL_RESIDUES) -> bool:
    """
    Convenience wrapper: True if ANY unnatural residue is present.
    """
    return len(find_unnatural_residues_in_smiles(smiles, patterns)) > 0
