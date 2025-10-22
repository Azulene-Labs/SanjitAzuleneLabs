
import joblib
import os

# Correct import for multi-instance prediction:
from tdc.multi_pred import DTI

import pandas as pd
import numpy as np
import sklearn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score



import numpy as np

def convert_to_dG(row):
    # BindingDB Y is in nM — convert to molar
    kd_molar = row["Y"] * 1e-9
    if kd_molar <= 0:
        return np.nan
    R = 0.001987  # kcal/(mol*K)
    T = 298
    return -R * T * np.log(kd_molar)

from rdkit import Chem
from rdkit.Chem import Descriptors

def compute_rdkit_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }



# Then, access the specific BindingDB dataset by name
data = DTI(name='BindingDB_Kd')  # For datasets with Kd units
df = data.get_data()

df["binding_free_energy"] = df.apply(convert_to_dG, axis=1)
df = df.dropna(subset=["binding_free_energy", "Drug"])


tqdm.pandas()
from rdkit import Chem

valid_rows = []
features = []

for idx, smiles in tqdm(df["Drug"].items(), desc="Computing RDKit features"):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        continue  # skip invalid SMILES
    try:
        feat = compute_rdkit_features(smiles)
        if feat is not None:
            features.append(feat)
            valid_rows.append(idx)
    except Exception:
        continue  # skip problematic rows

# Subset df to only valid molecules
df_valid = df.loc[valid_rows].reset_index(drop=True)
features_df = pd.DataFrame(features).reset_index(drop=True)

# Merge cleanly
df = pd.concat([df_valid, features_df], axis=1)
print(f"✅ Computed RDKit features for {len(df)} valid molecules (skipped {len(df_valid) - len(df)}).")



X = df[["MolWt", "LogP", "TPSA", "NumHDonors", "NumHAcceptors", "RotBonds"]]
y = df["binding_free_energy"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



rf = RandomForestRegressor(n_estimators=400, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(f"R² = {r2_score(y_test, y_pred):.3f}")
print(f"MAE = {mean_absolute_error(y_test, y_pred):.3f} kcal/mol")



os.makedirs("models", exist_ok=True)
joblib.dump(rf, "models/binding_free_energy_rf.joblib")