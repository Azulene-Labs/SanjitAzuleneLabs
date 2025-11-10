import pytest
from rdkit import Chem
from search_uaa import contains_uaa
    

@pytest.fixture
def uaa_smiles():
    # example from iNClusive (change to real UAA you choose)
    return "O=C(O)[C@H](Cc1ccccc1)N"  # unnatural Phe variant

def test_exact_match(uaa_smiles):
    assert contains_uaa(uaa_smiles, uaa_smiles) is True

def test_present_in_peptide(uaa_smiles):
    peptide = "CC(=O)N[C@@H](Cc1ccccc1)C(=O)NCC"  # same monomer interior
    assert contains_uaa(peptide, uaa_smiles) is True

def test_not_present(uaa_smiles):
    gly = "NCC(=O)O"
    assert contains_uaa(gly, uaa_smiles) is False

def test_handles_invalid_smiles(uaa_smiles):
    assert contains_uaa("NOT_SMILES", uaa_smiles) is False