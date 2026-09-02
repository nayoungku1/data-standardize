"""
verify_standardized.py
----------------------
Validates that standardized h5ad files have been processed correctly:
  1. Checks if all files exist
  2. Verifies that gene lists in each file are a subset of the standard reference (gene_names.csv)
  3. Confirms gene order matches the relative standard order
  4. Detects any missing genes (genes that should overlap but are absent) or extra genes
  5. Verifies all datasets defined in datasets_meta.json are processed
"""

import json
import os
import sys
import pandas as pd
import anndata as ad
import numpy as np

META_JSON   = "/mnt/nas2/projects/vcc-data/datasets_meta.json"
GENE_CSV    = "/mnt/nas2/projects/vcc-data/vcc-2026/gene_names.csv"
OUTPUT_DIR  = "/mnt/nas2/projects/vcc-data/standardized"

print("=" * 70)
print("VCC Standardized H5AD Validation Script")
print("=" * 70)

# 1. Load standard gene list
std_genes_df = pd.read_csv(GENE_CSV)
std_genes    = std_genes_df.iloc[:, 0].dropna().astype(str).tolist()
std_genes_set = set(std_genes)
std_rank     = {g: i for i, g in enumerate(std_genes)}
print(f"\n[Standard Genes] Total {len(std_genes):,} genes (gene_names.csv)")

# 2. Load datasets_meta.json
with open(META_JSON) as f:
    meta = json.load(f)

print(f"[Metadata JSON] {len(meta)} datasets: {list(meta.keys())}")

# 3. Validate each dataset
summary_rows = []
all_ok = True

for name, entry in meta.items():
    out_path = os.path.join(OUTPUT_DIR, f"{name}_standardized.h5ad")
    
    row = {
        "dataset": name,
        "file_exists": os.path.exists(out_path),
        "n_cells": None,
        "n_overlap_genes": None,
        "expected_overlap": None,
        "gene_order_ok": None,
        "is_subset_of_std": None,
        "missing_genes": None,
        "extra_genes": None,
        "issues": []
    }
    
    if not row["file_exists"]:
        row["issues"].append("File not found!")
        all_ok = False
        summary_rows.append(row)
        continue
    
    # Load backed mode to save memory
    try:
        adata = ad.read_h5ad(out_path, backed="r")
    except Exception as e:
        row["issues"].append(f"File read error: {e}")
        all_ok = False
        summary_rows.append(row)
        continue
    
    file_genes = adata.var_names.tolist()
    n_cells    = adata.n_obs
    n_genes    = adata.n_vars
    adata.file.close()
    
    row["n_cells"] = n_cells
    row["n_overlap_genes"] = n_genes
    
    # 4. Calculate expected overlap from source via gene_conversion
    conv = entry.get("gene_conversion", {g: g for g in entry["gene"]})
    src_genes = entry["gene"]
    converted = [conv.get(g, g) for g in src_genes]
    
    seen = set()
    converted_unique = []
    for c in converted:
        if c not in seen:
            converted_unique.append(c)
            seen.add(c)
    
    expected_overlap = [g for g in std_genes if g in set(converted_unique)]
    row["expected_overlap"] = len(expected_overlap)
    
    file_genes_set    = set(file_genes)
    expected_set      = set(expected_overlap)
    
    # 5. Check consistency
    missing = expected_set - file_genes_set
    extra   = file_genes_set - expected_set
    not_in_std = file_genes_set - std_genes_set
    
    row["is_subset_of_std"] = len(not_in_std) == 0
    row["missing_genes"]    = len(missing)
    row["extra_genes"]      = len(extra)
    
    if missing:
        row["issues"].append(f"Missing {len(missing)} genes: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
        all_ok = False
    if extra:
        row["issues"].append(f"Extra {len(extra)} genes: {sorted(extra)[:10]}{'...' if len(extra)>10 else ''}")
        all_ok = False
    if not_in_std:
        row["issues"].append(f"Non-standard genes present: {len(not_in_std)}")
        all_ok = False
    
    # 6. Check relative gene order (strictly increasing based on standard reference)
    file_ranks = [std_rank.get(g, -1) for g in file_genes if g in std_rank]
    if len(file_ranks) > 1:
        is_sorted = all(file_ranks[i] < file_ranks[i+1] for i in range(len(file_ranks)-1))
        row["gene_order_ok"] = is_sorted
        if not is_sorted:
            row["issues"].append("Gene ordering mismatch against reference!")
            all_ok = False
    else:
        row["gene_order_ok"] = True
    
    # 7. Check gene count match
    if n_genes != len(expected_overlap):
        row["issues"].append(f"Gene count mismatch: file={n_genes}, expected={len(expected_overlap)}")
        all_ok = False
    
    summary_rows.append(row)

# Print Summary
print("\n" + "=" * 70)
print("Validation Results Summary")
print("=" * 70)

print(f"\n{'Dataset':<35} {'File':<5} {'Cells':>8} {'Genes':>8} {'Expected':>8} {'Order':>5} {'Issues'}")
print("-" * 80)

for r in summary_rows:
    exists_mark = "OK" if r["file_exists"] else "NO"
    order_mark  = ("OK" if r["gene_order_ok"] else "NG") if r["gene_order_ok"] is not None else "-"
    issue_str   = " | ".join(r["issues"]) if r["issues"] else "OK"
    
    print(f"{r['dataset']:<35} {exists_mark:<5} "
          f"{str(r['n_cells'] or '-'):>8} "
          f"{str(r['n_overlap_genes'] or '-'):>8} "
          f"{str(r['expected_overlap'] or '-'):>8} "
          f"{order_mark:>5}  {issue_str}")

print("\n" + "=" * 70)
if all_ok:
    print("ALL PASS: All datasets successfully validated!")
else:
    print("FAIL: Some datasets have issues! Review issues above.")
print("=" * 70)

# Detailed Missing Gene Analysis
print("\n\n[Detailed] Dataset Missing Gene Analysis")
print("-" * 70)

for name, entry in meta.items():
    out_path = os.path.join(OUTPUT_DIR, f"{name}_standardized.h5ad")
    if not os.path.exists(out_path):
        continue
    
    try:
        adata = ad.read_h5ad(out_path, backed="r")
        file_genes = set(adata.var_names.tolist())
        adata.file.close()
    except:
        continue
    
    conv = entry.get("gene_conversion", {g: g for g in entry["gene"]})
    src_genes = entry["gene"]
    converted = [conv.get(g, g) for g in src_genes]
    
    seen = set()
    converted_unique = []
    for c in converted:
        if c not in seen:
            converted_unique.append(c)
            seen.add(c)
    
    expected_overlap = set(g for g in std_genes if g in set(converted_unique))
    missing = expected_overlap - file_genes
    
    src_genes_in_std = set(converted_unique) & std_genes_set
    src_genes_NOT_in_std = set(converted_unique) - std_genes_set
    
    print(f"\n[{name}]")
    print(f"  Source gene count: {len(src_genes):,}  ->  Converted unique: {len(converted_unique):,}")
    print(f"  Overlapping standard genes: {len(src_genes_in_std):,}  (Non-standard genes: {len(src_genes_NOT_in_std):,})")
    print(f"  Saved file gene count: {len(file_genes):,}")
    if missing:
        print(f"  WARNING: Missing {len(missing)} genes: {sorted(missing)[:20]}")
    else:
        print(f"  OK: Zero missing genes")

print("\nValidation finished.")
