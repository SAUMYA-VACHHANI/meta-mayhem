"""
Fetch selected datasets from OpenML, save in local structured dataset repository,
and build dataset_catalog.csv.
"""

import openml
import os
import json
import pandas as pd

# ⚙️ Settings
OUTPUT_DIR = "datasets"
MAX_ROWS = 100_000  # skip huge datasets for practical speed

# 🧩 Selected OpenML dataset IDs (balanced classification & regression)
dataset_ids = [
     3,    6,   12,   14,   15,   16,   18,   21,   22,   23,   24,   25,   28,   29,   31,   32,   33,   37,
    44,   46,   49,   50,   54,   60,   61,   111,  146,  151,  155,  156,  180,  181,  182,  183,  184,  185,
   187,  188,  189,  190,  293,  294,  300,  307,  312,  313,  333,  334,  335,  336,  337,  338,  339,  340,
   341,  342,  348,  351,  354,  357,  360,  361,  375,  377,  378,  379,  380,  381,  382,  383,  384,  385,
   386,  387,  388,  389,  391,  392,  394,  395,  396,  397,  398,  400,  401,  405,  406,  409,  410,  
   411,  412,  413,  414,  415,  416,  417,  418
]


# 📦 Make base directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

catalog_rows = []

print(f"Fetching {len(dataset_ids)} datasets from OpenML...")

for did in dataset_ids:
    try:
        dataset = openml.datasets.get_dataset(did)
        df, *_ = dataset.get_data()
        
        if len(df) > MAX_ROWS:
            print(f"Skipping {dataset.name} (ID {did}): too large ({len(df)} rows)")
            continue

        dataset_id = f"openml_{did}"
        folder = os.path.join(OUTPUT_DIR, dataset_id)
        os.makedirs(folder, exist_ok=True)

        # Save data.csv
        csv_path = os.path.join(folder, "data.csv")
        df.to_csv(csv_path, index=False)

        # Save metadata.json
        meta = {
            "dataset_id": dataset_id,
            "original_id": did,
            "name": dataset.name,
            "n_rows": len(df),
            "n_columns": df.shape[1],
            "default_target_attribute": dataset.default_target_attribute,
            "source": "OpenML",
            "url": dataset.url,
            "licence": dataset.licence
        }
        json_path = os.path.join(folder, "metadata.json")
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Add to catalog
        catalog_rows.append({
            "dataset_id": dataset_id,
            "file_path": os.path.relpath(csv_path, OUTPUT_DIR),
            "task_type": "classification",  # as default; update manually for regression datasets
            "target_column": dataset.default_target_attribute,
            "source": "OpenML",
            "notes": ""
        })

        print(f"Saved {dataset.name} (ID {did}) with {len(df)} rows")

    except Exception as e:
        print(f"Failed to fetch dataset ID {did}: {e}")

# 📋 Build dataset_catalog.csv
catalog_df = pd.DataFrame(catalog_rows)
catalog_csv = os.path.join(OUTPUT_DIR, "dataset_catalog.csv")
catalog_df.to_csv(catalog_csv, index=False)

print(f"\nDone! Catalog saved to {catalog_csv}")
print(f"Total datasets saved: {len(catalog_rows)}")