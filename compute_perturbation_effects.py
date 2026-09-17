#!/usr/bin/env python3
"""
compute_perturbation_effects.py
-------------------------------
Computes perturbation-specific effect metrics (logFC and delta) across standardized
h5ad datasets relative to the non-targeting control ('non-targeting') and stores them in adata.uns.

Metrics computed:
  1. Normalized (Option A):
     - Each cell is library-size normalized to 10,000 counts (CP10k) and log1p transformed:
       Y = ln(1 + (X / sum(X)) * 10,000)
     - delta_norm = mean(Y_pert) - mean(Y_ctrl)
     - logfc_norm = log2(e) * delta_norm  (equivalent to log2 fold change)
  2. Raw Counts (Option B):
     - Computed directly on raw integer expression matrix X:
       mu_pert = mean(X_pert), mu_ctrl = mean(X_ctrl)
     - delta_raw = mu_pert - mu_ctrl
     - logfc_raw = log2((mu_pert + eps) / (mu_ctrl + eps))  (default eps = 0.1)
  3. Aliases:
     - logfc -> logfc_norm
     - delta -> delta_norm

Storage in adata.uns:
  - Stored directly into adata.uns via h5py (r+ mode):
    - uns['logfc_norm'] : 2D float32 array (shape: n_perturbations x n_genes)
    - uns['delta_norm'] : 2D float32 array (shape: n_perturbations x n_genes)
    - uns['logfc_raw']  : 2D float32 array (shape: n_perturbations x n_genes)
    - uns['delta_raw']  : 2D float32 array (shape: n_perturbations x n_genes)
    - uns['logfc']      : alias pointing to logfc_norm
    - uns['delta']      : alias pointing to delta_norm
    - uns['perturbations'] : 1D string array of perturbation names (matching rows)
  - Zero rewriting of X, layers, obs, or var (instantaneous save < 0.2s).
  - Can be easily accessed as a DataFrame in Python via get_perturbation_effect(adata, 'logfc_norm').

Target Datasets (10 datasets, strictly excluding mixscale_*):
  1. kaggle
  2. nadig_hepg2
  3. nadig_jurkat
  4. replogle_rpe1
  5. replogle_k562_essential
  6. replogle_k562_gwp
  7. orion_hct116
  8. orion_hek293t
  9. arc_h1
  10. kolf_strong

Usage:
    # Dry-run calculation on a single dataset:
    python compute_perturbation_effects.py --dataset kaggle --dry-run

    # Compute and save in-place for a specific dataset:
    python compute_perturbation_effects.py --dataset kaggle

    # Batch process all 10 non-mixscale standardized datasets:
    python compute_perturbation_effects.py --all
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import logging
import time
import warnings
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore")

import sys
sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# Default directory
DEFAULT_STANDARDIZED_DIR = "/mnt/nas2/projects/vcc-data/standardized"

# Canonical non-targeting label and perturbation column
STD_PERT_COL = "target_gene"
STD_CTRL_LABEL = "non-targeting"

# Target datasets (10 datasets, strictly excluding mixscale_*)
DEFAULT_TARGET_DATASETS = [
    "kaggle",
    "nadig_hepg2",
    "nadig_jurkat",
    "replogle_rpe1",
    "replogle_k562_essential",
    "replogle_k562_gwp",
    "orion_hct116",
    "orion_hek293t",
    "arc_h1",
    "kolf_strong",
]

PRISM_TARGET_DATASETS = [
    "prism_gse210681",
    "prism_gse221321",
    "prism_gse208240",
    "prism_gse150062",
    "prism_gse261283",
    "prism_gse165291",
]

EXCLUDED_DATASETS = {
    "mixscale_ifnb",
    "mixscale_ifng",
    "mixscale_ins",
    "mixscale_tgfb",
    "mixscale_tnfa",
}


def get_var_gene_names(h5f: h5py.File) -> List[str]:
    """Extracts gene names from the var group in AnnData h5ad."""
    if "var" not in h5f:
        raise ValueError("Missing 'var' group in h5ad file")
    
    var_grp = h5f["var"]
    
    # Check _index
    if "_index" in var_grp:
        raw_idx = var_grp["_index"][:]
        return [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in raw_idx]
    
    # Check gene_name
    if "gene_name" in var_grp:
        col = var_grp["gene_name"]
        if isinstance(col, h5py.Group) and "categories" in col and "codes" in col:
            cats = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in col["categories"][:]]
            codes = col["codes"][:]
            return [cats[c] if c >= 0 else "" for c in codes]
        else:
            return [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in col[:]]
            
    if "features" in var_grp:
        return [g.decode("utf-8") if isinstance(g, bytes) else str(g) for g in var_grp["features"][:]]
        
    raise ValueError(f"Cannot identify gene names in var (keys: {list(var_grp.keys())})")


def get_obs_perturbations(h5f: h5py.File, pert_col: str) -> np.ndarray:
    """Extracts perturbation string array from obs[pert_col]."""
    if "obs" not in h5f or pert_col not in h5f["obs"]:
        raise ValueError(f"'{pert_col}' not found in obs group")
        
    col = h5f["obs"][pert_col]
    if isinstance(col, h5py.Group) and "categories" in col and "codes" in col:
        cats = [c.decode("utf-8") if isinstance(c, bytes) else str(c) for c in col["categories"][:]]
        codes = col["codes"][:]
        return np.array([cats[c] if c >= 0 else "nan" for c in codes])
    else:
        raw_vals = col[:]
        return np.array([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw_vals])


def compute_effects(
    h5ad_path: str,
    chunk_size: int = 50000,
    eps: float = 0.1,
    target_sum: float = 10000.0,
    pert_col: str = STD_PERT_COL,
    ctrl_label: str = STD_CTRL_LABEL,
) -> Dict[str, Union[np.ndarray, List[str], int, pd.Series]]:
    """
    Computes delta and logfc for both normalized and raw expression
    using chunked streaming accumulation over h5py.
    """
    t0 = time.time()
    log.info(f"Opening file: {h5ad_path}")
    
    with h5py.File(h5ad_path, "r") as f:
        gene_names = get_var_gene_names(f)
        n_genes = len(gene_names)
        
        perts_all = get_obs_perturbations(f, pert_col)
        n_cells = len(perts_all)
        log.info(f"  Total cells: {n_cells:,} | Total genes: {n_genes:,}")
        
        # Verify control
        is_ctrl = (perts_all == ctrl_label)
        n_ctrl = int(is_ctrl.sum())
        if n_ctrl == 0:
            raise ValueError(f"No control cells found with label '{ctrl_label}' in obs['{pert_col}']")
        log.info(f"  Control cells ('{ctrl_label}'): {n_ctrl:,} ({n_ctrl / n_cells * 100:.2f}%)")
        
        # Unique perturbations (excluding control and nan)
        unique_perts = sorted([p for p in np.unique(perts_all) if p != ctrl_label and p != "nan"])
        n_perts = len(unique_perts)
        log.info(f"  Unique non-control perturbations: {n_perts:,}")
        
        pert_to_idx = {p: i for i, p in enumerate(unique_perts)}
        
        # Initialize accumulators in double precision
        norm_sum_perts = np.zeros((n_perts, n_genes), dtype=np.float64)
        norm_sum_ctrl = np.zeros(n_genes, dtype=np.float64)
        
        raw_sum_perts = np.zeros((n_perts, n_genes), dtype=np.float64)
        raw_sum_ctrl = np.zeros(n_genes, dtype=np.float64)
        
        pert_counts = np.zeros(n_perts, dtype=np.int64)
        
        # Inspect X matrix
        if "X" not in f:
            raise ValueError("Missing 'X' group/dataset in h5ad file")
            
        x_obj = f["X"]
        is_csr = isinstance(x_obj, h5py.Group) and "data" in x_obj and "indptr" in x_obj
        is_dense = isinstance(x_obj, h5py.Dataset)
        
        if not (is_csr or is_dense):
            raise NotImplementedError(f"Unsupported X format: {type(x_obj)}")
            
        log.info(f"  Streaming through X (chunk_size={chunk_size:,})...")
        
        if is_csr:
            indptr_ds = x_obj["indptr"]
            indices_ds = x_obj["indices"]
            data_ds = x_obj["data"]
            
            for start_idx in range(0, n_cells, chunk_size):
                end_idx = min(start_idx + chunk_size, n_cells)
                
                c_indptr = indptr_ds[start_idx : end_idx + 1]
                data_start = c_indptr[0]
                data_end = c_indptr[-1]
                
                chunk_data = data_ds[data_start:data_end]
                chunk_indices = indices_ds[data_start:data_end]
                shifted_indptr = c_indptr - data_start
                
                X_chunk = sp.csr_matrix(
                    (chunk_data, chunk_indices, shifted_indptr),
                    shape=(end_idx - start_idx, n_genes),
                    dtype=np.float32
                )
                
                # 1. Normalized: CP10k + log1p
                lib_sizes = np.asarray(X_chunk.sum(axis=1)).ravel()
                scale = np.zeros_like(lib_sizes, dtype=np.float32)
                valid_mask = lib_sizes > 0
                scale[valid_mask] = target_sum / lib_sizes[valid_mask]
                
                X_norm = X_chunk.multiply(scale[:, None]).tocsr()
                X_norm.data = np.log1p(X_norm.data)
                
                # Chunk perturbations
                chunk_perts = perts_all[start_idx:end_idx]
                
                # Accumulate control
                ctrl_mask = (chunk_perts == ctrl_label)
                if np.any(ctrl_mask):
                    sub_raw = X_chunk[ctrl_mask]
                    sub_norm = X_norm[ctrl_mask]
                    raw_sum_ctrl += np.asarray(sub_raw.sum(axis=0)).ravel()
                    norm_sum_ctrl += np.asarray(sub_norm.sum(axis=0)).ravel()
                    
                # Accumulate perturbations
                chunk_unique = np.unique(chunk_perts)
                for p in chunk_unique:
                    if p == ctrl_label or p == "nan":
                        continue
                    p_idx = pert_to_idx[p]
                    p_mask = (chunk_perts == p)
                    pert_counts[p_idx] += int(p_mask.sum())
                    
                    sub_raw = X_chunk[p_mask]
                    sub_norm = X_norm[p_mask]
                    raw_sum_perts[p_idx] += np.asarray(sub_raw.sum(axis=0)).ravel()
                    norm_sum_perts[p_idx] += np.asarray(sub_norm.sum(axis=0)).ravel()
                    
                if (end_idx // chunk_size) % 5 == 0 or end_idx == n_cells:
                    pct = (end_idx / n_cells) * 100
                    elapsed_chunk = time.time() - t0
                    log.info(f"    Processed {end_idx:,} / {n_cells:,} cells ({pct:.1f}%) in {elapsed_chunk:.1f}s")
                    
        elif is_dense:
            for start_idx in range(0, n_cells, chunk_size):
                end_idx = min(start_idx + chunk_size, n_cells)
                X_chunk = np.array(x_obj[start_idx:end_idx], dtype=np.float32)
                
                lib_sizes = X_chunk.sum(axis=1)
                scale = np.zeros_like(lib_sizes, dtype=np.float32)
                valid_mask = lib_sizes > 0
                scale[valid_mask] = target_sum / lib_sizes[valid_mask]
                
                X_norm = np.log1p(X_chunk * scale[:, None])
                chunk_perts = perts_all[start_idx:end_idx]
                
                ctrl_mask = (chunk_perts == ctrl_label)
                if np.any(ctrl_mask):
                    raw_sum_ctrl += X_chunk[ctrl_mask].sum(axis=0)
                    norm_sum_ctrl += X_norm[ctrl_mask].sum(axis=0)
                    
                for p in np.unique(chunk_perts):
                    if p == ctrl_label or p == "nan":
                        continue
                    p_idx = pert_to_idx[p]
                    p_mask = (chunk_perts == p)
                    pert_counts[p_idx] += int(p_mask.sum())
                    raw_sum_perts[p_idx] += X_chunk[p_mask].sum(axis=0)
                    norm_sum_perts[p_idx] += X_norm[p_mask].sum(axis=0)

    # Compute Means
    ctrl_mean_norm = norm_sum_ctrl / n_ctrl
    ctrl_mean_raw = raw_sum_ctrl / n_ctrl
    
    valid_perts = pert_counts > 0
    pert_mean_norm = np.zeros_like(norm_sum_perts)
    pert_mean_raw = np.zeros_like(raw_sum_perts)
    
    pert_mean_norm[valid_perts] = norm_sum_perts[valid_perts] / pert_counts[valid_perts, None]
    pert_mean_raw[valid_perts] = raw_sum_perts[valid_perts] / pert_counts[valid_perts, None]
    
    # Metric 1: Normalized (Option A)
    log2_e = np.log2(np.e)
    delta_norm = (pert_mean_norm - ctrl_mean_norm[None, :]).astype(np.float32)
    logfc_norm = (delta_norm * log2_e).astype(np.float32)
    
    # Metric 2: Raw (Option B)
    delta_raw = (pert_mean_raw - ctrl_mean_raw[None, :]).astype(np.float32)
    logfc_raw = np.log2((pert_mean_raw + eps) / (ctrl_mean_raw[None, :] + eps)).astype(np.float32)
    
    elapsed = time.time() - t0
    log.info(f"  Computation finished successfully in {elapsed:.1f}s")
    
    return {
        "delta_norm": delta_norm,
        "logfc_norm": logfc_norm,
        "delta_raw": delta_raw,
        "logfc_raw": logfc_raw,
        "perturbations": unique_perts,
        "genes": gene_names,
        "n_ctrl": n_ctrl,
        "pert_counts": pd.Series(pert_counts, index=unique_perts),
    }


def write_effects_to_uns(h5ad_path: str, effects: dict, overwrite: bool = True):
    """
    Writes the computed effect matrices and perturbation labels directly into uns group in h5ad file.
    Uses 2D numpy arrays and 1D string array, ensuring single-dataset writing without metadata overhead.
    """
    t0 = time.time()
    log.info(f"Writing effect matrices to {h5ad_path} ['uns']...")
    
    perts_arr = np.array(effects["perturbations"], dtype=object)
    str_dtype = h5py.string_dtype(encoding="utf-8")
    
    with h5py.File(h5ad_path, "r+") as f:
        if "uns" not in f:
            f.create_group("uns")
        uns_grp = f["uns"]
        
        # Datasets to write
        datasets = {
            "logfc_norm": effects["logfc_norm"],
            "delta_norm": effects["delta_norm"],
            "logfc_raw":  effects["logfc_raw"],
            "delta_raw":  effects["delta_raw"],
            "logfc":      effects["logfc_norm"],  # Standard alias
            "delta":      effects["delta_norm"],  # Standard alias
        }
        
        for key, arr in datasets.items():
            if key in uns_grp:
                if overwrite:
                    del uns_grp[key]
                else:
                    log.warning(f"  Key '{key}' already in uns, skipping (use overwrite=True to replace)")
                    continue
            ds = uns_grp.create_dataset(key, data=arr, compression="gzip", compression_opts=4)
            ds.attrs["encoding-type"] = "array"
            ds.attrs["encoding-version"] = "0.2.0"
            
        # Write perturbation labels
        if "perturbations" in uns_grp:
            if overwrite:
                del uns_grp["perturbations"]
        p_ds = uns_grp.create_dataset("perturbations", data=perts_arr, dtype=str_dtype)
        p_ds.attrs["encoding-type"] = "string-array"
        p_ds.attrs["encoding-version"] = "0.2.0"
            
    elapsed = time.time() - t0
    log.info(f"  Successfully wrote 6 effect matrices and 'perturbations' array to uns in {elapsed:.2f}s!")


def get_perturbation_effect(
    adata,
    metric: str = "logfc_norm",
) -> pd.DataFrame:
    """
    Convenience getter for AnnData objects:
    Retrieves the 2D effect matrix from uns and wraps it as a labeled pandas DataFrame.
    
    Usage:
        df_lfc = get_perturbation_effect(adata, 'logfc_norm')
        df_lfc.loc['STAT1', 'ISG15']  # Direct label indexing
    """
    if metric not in adata.uns:
        raise KeyError(f"'{metric}' not found in adata.uns. Available: {list(adata.uns.keys())}")
        
    mat = adata.uns[metric]
    if isinstance(mat, pd.DataFrame):
        return mat
        
    if "perturbations" not in adata.uns:
        raise KeyError("'perturbations' list not found in adata.uns")
        
    perts = adata.uns["perturbations"]
    return pd.DataFrame(mat, index=perts, columns=adata.var_names)


def process_dataset(
    dataset_name: str,
    standardized_dir: str = DEFAULT_STANDARDIZED_DIR,
    chunk_size: int = 50000,
    eps: float = 0.1,
    target_sum: float = 10000.0,
    dry_run: bool = False,
    overwrite: bool = True,
) -> dict:
    """Processes a single dataset by name."""
    if dataset_name in EXCLUDED_DATASETS:
        log.warning(f"[{dataset_name}] Dataset is in EXCLUDED_DATASETS list. Skipping.")
        return {"dataset": dataset_name, "status": "excluded"}
        
    h5ad_path = os.path.join(standardized_dir, f"{dataset_name}_standardized.h5ad")
    if not os.path.exists(h5ad_path):
        log.error(f"[{dataset_name}] File not found: {h5ad_path}")
        return {"dataset": dataset_name, "status": "not_found"}
        
    log.info(f"\n=======================================================")
    log.info(f"Processing dataset: [{dataset_name}]")
    log.info(f"Path: {h5ad_path}")
    log.info(f"=======================================================")
    
    t0 = time.time()
    try:
        effects = compute_effects(
            h5ad_path=h5ad_path,
            chunk_size=chunk_size,
            eps=eps,
            target_sum=target_sum,
        )
        
        n_perts = len(effects["perturbations"])
        n_genes = len(effects["genes"])
        log.info(f"  Result shapes: perts={n_perts:,}, genes={n_genes:,}")
        
        # Self-target knockdown check
        gene_set = set(effects["genes"])
        gene_to_idx = {g: i for i, g in enumerate(effects["genes"])}
        pert_to_idx = {p: i for i, p in enumerate(effects["perturbations"])}
        
        self_hits = [p for p in effects["perturbations"][:15] if p in gene_set]
        if self_hits:
            log.info(f"  Sample target self-effects:")
            for p in self_hits[:3]:
                pi = pert_to_idx[p]
                gi = gene_to_idx[p]
                lfc_n = effects["logfc_norm"][pi, gi]
                del_n = effects["delta_norm"][pi, gi]
                lfc_r = effects["logfc_raw"][pi, gi]
                del_r = effects["delta_raw"][pi, gi]
                log.info(
                    f"    {p}: logfc_norm={lfc_n:+.3f}, "
                    f"delta_norm={del_n:+.3f}, "
                    f"logfc_raw={lfc_r:+.3f}, "
                    f"delta_raw={del_r:+.3f}"
                )
                
        if dry_run:
            log.info("  [DRY-RUN] Skipped writing to h5ad.")
            return {"dataset": dataset_name, "status": "dry_run", "elapsed": round(time.time() - t0, 1), "perts": n_perts}
            
        write_effects_to_uns(h5ad_path, effects, overwrite=overwrite)
        elapsed = time.time() - t0
        return {"dataset": dataset_name, "status": "done", "elapsed": round(elapsed, 1), "perts": n_perts}
        
    except Exception as e:
        log.error(f"[{dataset_name}] Error during processing: {e}", exc_info=True)
        return {"dataset": dataset_name, "status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Compute perturbation logFC and delta into adata.uns")
    parser.add_argument("--dataset", type=str, help="Specific dataset to process")
    parser.add_argument("--prism", action="store_true", help="Process all PRISM standardized datasets")
    parser.add_argument("--all", action="store_true", help="Process all target datasets (excluding mixscale_*)")
    parser.add_argument("--dir", type=str, default=DEFAULT_STANDARDIZED_DIR, help="Standardized directory")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for streaming calculation")
    parser.add_argument("--eps", type=float, default=0.1, help="Pseudo-count for raw logfc (default: 0.1)")
    parser.add_argument("--target-sum", type=float, default=10000.0, help="Normalization scale (default: 10000.0)")
    parser.add_argument("--dry-run", action="store_true", help="Calculate without modifying h5ad files")
    parser.add_argument("--overwrite", action="store_true", default=True, help="Overwrite existing uns keys")
    args = parser.parse_args()

    if not args.dataset and not args.all and not args.prism:
        parser.print_help()
        print("\nPlease specify either --dataset <name>, --prism, or --all")
        return

    datasets_to_process = []
    if args.dataset:
        datasets_to_process = [args.dataset]
    elif args.prism:
        datasets_to_process = PRISM_TARGET_DATASETS
    elif args.all:
        datasets_to_process = DEFAULT_TARGET_DATASETS

    log.info(f"Target datasets ({len(datasets_to_process)}): {datasets_to_process}")
    log.info(f"Directory: {args.dir}")
    log.info(f"Dry-run: {args.dry_run}")
    
    results = []
    total_t0 = time.time()
    
    for ds in datasets_to_process:
        res = process_dataset(
            dataset_name=ds,
            standardized_dir=args.dir,
            chunk_size=args.chunk_size,
            eps=args.eps,
            target_sum=args.target_sum,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        results.append(res)
        
    total_elapsed = time.time() - total_t0
    log.info("\n" + "=" * 65)
    log.info(f"Batch Processing Summary (Total time: {total_elapsed:.1f}s)")
    log.info("=" * 65)
    log.info(f"{'Dataset':<26} {'Status':<10} {'Perts':>8} {'Time (s)':>10}")
    log.info("-" * 65)
    for r in results:
        perts_str = str(r.get("perts", "-"))
        elapsed_str = str(r.get("elapsed", "-"))
        log.info(f"{r['dataset']:<26} {r['status']:<10} {perts_str:>8} {elapsed_str:>10}")


if __name__ == "__main__":
    main()
