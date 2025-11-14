import requests
import time
import csv
import pandas as pd

# Rate control (polite)
SLEEP = 0.25  # seconds between requests

seed_names = [
    "p-Azido-L-phenylalanine",
    "p-Azido-L-phenylalanine (pAzF)",
    "p-Acetyl-L-phenylalanine",
    "4-Acetyl-L-phenylalanine",
    "p-Benzoyl-L-phenylalanine",
    "4-Benzoyl-L-phenylalanine",
    "p-Boronophenylalanine",
    "p-Cyano-L-phenylalanine",
    "p-cyano-L-phenylalanine",
    "p-Azidomethyl-L-phenylalanine",
    "Azidohomoalanine",
    "L-Azidohomoalanine",
    "Homopropargylglycine",
    "L-homopropargylglycine",
    "O-Methyl-L-tyrosine",
    "O-methyl-L-tyrosine",
    "N-epsilon-propargyl-lysine",
    "Nε-propargyl-lysine",
    "N-epsilon-acetyl-lysine",
    "N-epsilon-acetyl-L-lysine",
    "p-Aminophenylalanine",
    "4-Bromophenylalanine",
    "p-Bromophenylalanine"
]

# PubChem PUG-REST endpoints
PUG_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

def get_cid_for_name(name):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PubChem-Client/1.0)"}
    url = f"{PUG_BASE}/compound/name/{requests.utils.requote_uri(name)}/cids/JSON"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            return None
        return cids[0]
    except Exception:
        return None


def get_props_for_cid(cid):
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PubChem-Client/1.0)"}
    props = ["MolecularFormula","MolecularWeight","IUPACName","CanonicalSMILES","InChIKey"]
    props_str = ",".join(props)

    url = f"{PUG_BASE}/compound/cid/{cid}/property/{props_str}/JSON"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        props_data = data.get("PropertyTable", {}).get("Properties", [])
        if not props_data:
            return None
        return props_data[0]
    except Exception:
        return None

def pubchem_url_for_cid(cid):
    return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"

def main():
    rows = []
    failed = []

    for nm in seed_names:
        print("Resolving:", nm)
        cid = get_cid_for_name(nm)
        time.sleep(SLEEP)
        if cid is None:
            print("  -> NO CID found for:", nm)
            failed.append(nm)
            continue
        print("  -> CID:", cid)
        props = get_props_for_cid(cid)
        time.sleep(SLEEP)
        if props is None:
            print("  -> Failed to fetch properties for CID:", cid)
            failed.append(nm)
            continue

        row = {
            "QueryName": nm,
            "CID": cid,
            "Name": props.get("IUPACName") or nm,
            "MolecularFormula": props.get("MolecularFormula"),
            "MolecularWeight": props.get("MolecularWeight"),
            "IUPACName": props.get("IUPACName"),
            "CanonicalSMILES": props.get("CanonicalSMILES"),
            "InChIKey": props.get("InChIKey"),
            "PubChemURL": pubchem_url_for_cid(cid),
        }
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        csv_out = "uaa_data/ncaa_pubchem_optionC.csv"
        df.to_csv(csv_out, index=False)
        print(f"Wrote {len(df)} records to {csv_out}")
    else:
        print("No records retrieved.")

    if failed:
        with open("uaa_data/failed_pubchem_queries.txt", "w") as fh:
            for q in failed:
                fh.write(q + "\n")
        print(f"{len(failed)} queries failed; see failed_pubchem_queries.txt")

if __name__ == "__main__":
    main()