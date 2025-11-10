# src/uaa_search.py
from rdkit import Chem

def contains_uaa(smiles: str, uaa_frag: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smiles)
        frag = Chem.MolFromSmiles(uaa_frag)
        if mol is None or frag is None:
            return False
        return mol.HasSubstructMatch(frag)
    except:
        return False
