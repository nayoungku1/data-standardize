#!/usr/bin/env python3
"""
filter_prism_crispri.py
-----------------------
Pipeline script for PRISM single-cell CRISPR datasets:
  1. Extracts only CRISPRi cells (filters out CRISPR KO, CRISPRn, CRISPRa, etc. via obs.crispr_type).
  2. Standardizes control cells to 'non-targeting' in perturbation_name based on condition / targeting columns.
  3. Filters out double / multiplex perturbations to retain ONLY single perturbations and non-targeting controls.
  4. Saves filtered AnnData to the target directory (default: /mnt/nas2/projects/vcc-data/standardized/).

Usage:
    # 1. Preview filtering statistics across target datasets (dry-run, instant)
    python filter_prism_crispri.py --dry-run

    # 2. Process a single dataset (dry-run)
    python filter_prism_crispri.py --dataset GSE221321 --dry-run

    # 3. Filter and save all target PRISM datasets
    python filter_prism_crispri.py

    # 4. Filter and overwrite existing files in custom directory
    python filter_prism_crispri.py --output-dir /mnt/nas2/projects/vcc-data/standardized/ --overwrite
"""

import argparse
import gc
import glob
import logging
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import scipy.sparse as sp

# Disable HDF5 file locking on NFS mounts
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np
import pandas as pd
import anndata as ad

# Import read_elem safely
try:
    from anndata.io import read_elem
except ImportError:
    from anndata._io import read_elem

# Enable AnnData nullable string writing
try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("filter_prism")

# Default paths
DEFAULT_INPUT_DIR = "/mnt/nas2/projects/vcc-data/PRISM/Datasets"
DEFAULT_OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"

# Target GSE datasets identified from PRISM-for-VCC2026.md
DEFAULT_TARGET_GSES = [
    "GSE210681",
    "GSE221321",
    "GSE208240",
    "GSE150062",
    "GSE261283",
    "GSE165291",
]

# Control labels in condition / targeting columns
CONTROL_CONDITIONS = {"control", "ctrl", "unperturbed", "negative_control", "negctrl"}
NON_TARGETING_LABELS = {
    "non-targeting", "non_targeting", "non targeting", "ntc", "nt",
    "control", "ctrl", "unperturbed", "safe_targeting", "aavs1", "clybl", "h11", "tigre"
}


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize pd.StringDtype arrays to object dtype to prevent AnnData write errors."""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) == "string":
            df[col] = df[col].astype(object).fillna("")
    return df


def load_obs_fast(h5ad_path: str) -> Tuple[pd.DataFrame, int]:
    """Fast load of obs and var dimension via h5py without loading X."""
    with h5py.File(h5ad_path, "r") as h5f:
        obs = read_elem(h5f["obs"])
        n_vars = 0
        if "var" in h5f:
            if "_index" in h5f["var"]:
                n_vars = len(h5f["var"]["_index"])
            elif "gene_name" in h5f["var"]:
                n_vars = len(h5f["var"]["gene_name"])
            elif "column-order" in h5f["var"].attrs and len(h5f["var"].attrs["column-order"]) > 0:
                first_col = h5f["var"].attrs["column-order"][0]
                n_vars = len(h5f["var"][first_col])
        if n_vars == 0:
            if "X" in h5f:
                if hasattr(h5f["X"], "shape") and len(h5f["X"].shape) > 1:
                    n_vars = h5f["X"].shape[1]
                elif "shape" in h5f["X"].attrs:
                    n_vars = int(h5f["X"].attrs["shape"][1])
        return obs, n_vars


def filter_crispri_mask(obs: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, int]]:
    """Step 1: Extract only CRISPRi cells."""
    n_total = len(obs)
    if "crispr_type" not in obs.columns:
        log.warning("  'crispr_type' column not found in obs. Assuming all cells are CRISPRi.")
        return np.ones(n_total, dtype=bool), {"CRISPRi": n_total}

    raw_counts = obs["crispr_type"].value_counts(dropna=False).to_dict()
    crispr_type_str = obs["crispr_type"].astype(str).str.strip().str.lower()
    mask = crispr_type_str.isin(["crispri", "crispr i"])
    return mask.values, raw_counts


def standardize_control_perturbations(obs: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Step 2: If condition or targeting indicates control, or if perturbation_name is missing/safe-harbor,
    replace perturbation_name with canonical 'non-targeting'.
    """
    obs = obs.copy()
    pert_col = "perturbation_name" if "perturbation_name" in obs.columns else None
    if pert_col is None:
        for cand in ["target_gene", "gene_target", "gene"]:
            if cand in obs.columns:
                pert_col = cand
                break

    if pert_col is None:
        log.warning("  No perturbation column found to standardize.")
        return obs, 0

    # Ensure column is object or categorical with 'non-targeting' category
    if isinstance(obs[pert_col].dtype, pd.CategoricalDtype):
        if "non-targeting" not in obs[pert_col].cat.categories:
            obs[pert_col] = obs[pert_col].cat.add_categories(["non-targeting"])

    # Identify control cells
    is_ctrl_cond = np.zeros(len(obs), dtype=bool)
    if "condition" in obs.columns:
        cond_str = obs["condition"].astype(str).str.strip().str.lower()
        is_ctrl_cond |= cond_str.isin(CONTROL_CONDITIONS)

    if "targeting" in obs.columns:
        targeting_str = obs["targeting"].astype(str).str.strip().str.lower()
        is_ctrl_cond |= targeting_str.isin(["non-targeting", "non_targeting", "non targeting"])

    # Identify cells where perturbation_name is NaN, null, or a known safe-harbor control while in control condition
    pert_str = obs[pert_col].astype(str).str.strip().str.lower()
    is_safe_harbor = pert_str.isin(NON_TARGETING_LABELS) | obs[pert_col].isna()

    ctrl_mask = is_ctrl_cond | (is_safe_harbor & is_ctrl_cond)

    # Track how many values will be modified
    needs_replace = ctrl_mask & (obs[pert_col].astype(str) != "non-targeting")
    n_replaced = int(needs_replace.sum())

    if n_replaced > 0:
        obs.loc[ctrl_mask, pert_col] = "non-targeting"

    return obs, n_replaced


def filter_single_perturbation_mask(
    obs: pd.DataFrame,
    method: str = "both"
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Step 3: Filter out double / multiplex perturbations.
    Retains ONLY single perturbation cells and non-targeting controls.

    Method options:
      - 'both': Check guide count > 1 OR combination delimiters in perturbation_name.
      - 'count': Check guide count > 1 only.
      - 'delimiter': Check combination delimiters only.
    """
    n_total = len(obs)
    pert_col = "perturbation_name" if "perturbation_name" in obs.columns else None
    if pert_col is None:
        for cand in ["target_gene", "gene_target", "gene"]:
            if cand in obs.columns:
                pert_col = cand
                break

    # Non-targeting controls are always preserved
    if pert_col is not None:
        is_control = (obs[pert_col].astype(str).str.strip().str.lower() == "non-targeting")
    else:
        is_control = np.zeros(n_total, dtype=bool)

    is_multiplex = np.zeros(n_total, dtype=bool)

    # 1. Guide/Feature count check
    count_cols = [c for c in ["n_guides", "number_of_guides", "Total_number_of_guides", "num_features"] if c in obs.columns]
    multiplex_by_count = np.zeros(n_total, dtype=bool)

    if method in ("both", "count") and count_cols:
        for c in count_cols:
            try:
                counts = pd.to_numeric(obs[c], errors="coerce").fillna(1)
                # Count > 1 indicates double/multiple guides/features
                multiplex_by_count |= (counts > 1).values
            except Exception as e:
                log.warning(f"  Failed to parse guide count column '{c}': {e}")

    # 2. Delimiter check on perturbation_name
    multiplex_by_delim = np.zeros(n_total, dtype=bool)
    if method in ("both", "delimiter") and pert_col is not None:
        pert_series = obs[pert_col].astype(str)
        delim_regex = r"(?:\s*\+\s*|\s*;\s*|\s*,\s*|--)"
        has_delim = pert_series.str.contains(delim_regex, regex=True)
        # Exclude non-targeting controls
        multiplex_by_delim = (has_delim & ~is_control).values

    # Combine multiplex flags (only for non-control cells)
    if method == "both":
        is_multiplex = (multiplex_by_count | multiplex_by_delim) & ~is_control
    elif method == "count":
        is_multiplex = multiplex_by_count & ~is_control
    elif method == "delimiter":
        is_multiplex = multiplex_by_delim & ~is_control

    keep_mask = ~is_multiplex

    stats = {
        "multiplex_by_count": int((multiplex_by_count & ~is_control).sum()),
        "multiplex_by_delim": int((multiplex_by_delim & ~is_control).sum()),
        "total_multiplex_removed": int(is_multiplex.sum()),
        "single_pert_retained": int((keep_mask & ~is_control).sum()),
        "control_retained": int(is_control.sum()),
    }
    return keep_mask, stats


class IncrementalCSRWriter:
    """Incrementally appends CSR sparse chunks directly to an h5py Group."""
    def __init__(self, h5grp: h5py.Group, n_cols: int, expected_cells: int, dtype=np.float32):
        self.grp = h5grp
        self.n_cols = n_cols
        self.dtype = dtype
        self._nnz = 0
        self._n_rows = 0

        self.ds_data = h5grp.create_dataset(
            "data", shape=(0,), maxshape=(None,), dtype=dtype,
            chunks=(1 << 20,), compression="gzip", compression_opts=4)
        self.ds_indices = h5grp.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(1 << 20,), compression="gzip", compression_opts=4)
        self.ds_indptr = h5grp.create_dataset(
            "indptr", shape=(1,), maxshape=(None,), dtype=np.int64,
            chunks=(1 << 16,), compression="gzip", compression_opts=4)
        self.ds_indptr[0] = 0

        h5grp.attrs["encoding-type"] = "csr_matrix"
        h5grp.attrs["encoding-version"] = "0.1.0"
        h5grp.attrs["shape"] = [expected_cells, n_cols]

    def append_chunk(self, mat: sp.csr_matrix):
        mat = mat.astype(self.dtype)
        mat.sort_indices()
        nnz = mat.nnz
        nr = mat.shape[0]

        if nnz > 0:
            o = self._nnz
            self.ds_data.resize((o + nnz,))
            self.ds_indices.resize((o + nnz,))
            self.ds_data[o:] = mat.data
            self.ds_indices[o:] = mat.indices.astype(np.int32)

        op = self._n_rows + 1
        new_iptr = mat.indptr[1:].astype(np.int64) + self._nnz
        self.ds_indptr.resize((op + nr,))
        self.ds_indptr[op: op + nr] = new_iptr

        self._nnz += nnz
        self._n_rows += nr

    def finalize(self):
        self.grp.attrs["shape"] = [self._n_rows, self.n_cols]


def stream_slice_X(
    h5src: h5py.File,
    h5dst: h5py.File,
    combined_mask: np.ndarray,
    n_vars: int,
    chunk_size: int = 25000,
):
    """
    Streamingly slices rows of X according to combined_mask and writes
    directly to h5dst['X'] as an AnnData-compatible CSR matrix.
    Avoids loading the full matrix into RAM or doing unbuffered random I/O over NFS.
    """
    x_src = h5src["X"]
    n_initial = len(combined_mask)
    n_kept = int(np.sum(combined_mask))

    if "X" in h5dst:
        del h5dst["X"]

    x_grp = h5dst.create_group("X")
    writer = IncrementalCSRWriter(x_grp, n_cols=n_vars, expected_cells=n_kept)

    if isinstance(x_src, h5py.Group):
        enc = x_src.attrs.get("encoding-type", "csr_matrix")
        if enc == "csr_matrix":
            indptr_arr = x_src["indptr"]
            indices_arr = x_src["indices"]
            data_arr = x_src["data"]

            for start in range(0, n_initial, chunk_size):
                end = min(start + chunk_size, n_initial)
                chunk_mask = combined_mask[start:end]
                n_sub = int(np.sum(chunk_mask))
                if n_sub == 0:
                    continue

                iptr = indptr_arr[start:end + 1].astype(np.int64)
                rs, re = int(iptr[0]), int(iptr[-1])

                if rs == re:
                    writer.append_chunk(sp.csr_matrix((n_sub, n_vars), dtype=np.float32))
                    continue

                c_data = data_arr[rs:re].astype(np.float32)
                c_indices = indices_arr[rs:re].astype(np.int32)
                iptr_local = iptr - rs

                if chunk_mask.all():
                    mat = sp.csr_matrix((c_data, c_indices, iptr_local), shape=(end - start, n_vars), dtype=np.float32)
                else:
                    nnz_per_row = np.diff(iptr_local)
                    row_ids = np.repeat(np.arange(end - start, dtype=np.int64), nnz_per_row)
                    keep_nnz = chunk_mask[row_ids]

                    row_map = np.full(end - start, -1, dtype=np.int64)
                    row_map[chunk_mask] = np.arange(n_sub, dtype=np.int64)

                    new_rows = row_map[row_ids[keep_nnz]]
                    new_cols = c_indices[keep_nnz]
                    new_vals = c_data[keep_nnz]
                    mat = sp.csr_matrix((new_vals, (new_rows, new_cols)), shape=(n_sub, n_vars), dtype=np.float32)

                writer.append_chunk(mat)
                del c_data, c_indices, iptr_local, mat
                gc.collect()

                if start % (chunk_size * 4) == 0 or end == n_initial:
                    log.info(f"    Exporting X (CSR): {end:,}/{n_initial:,} rows processed ({writer._n_rows:,} cells retained)")

            writer.finalize()

        elif enc == "csc_matrix":
            indptr_arr = x_src["indptr"]
            indices_arr = x_src["indices"]
            data_arr = x_src["data"]
            n_cols = len(indptr_arr) - 1

            row_map = np.full(n_initial, -1, dtype=np.int32)
            row_map[combined_mask] = np.arange(n_kept, dtype=np.int32)

            all_rows, all_cols, all_vals = [], [], []
            for col_idx in range(n_cols):
                cs = int(indptr_arr[col_idx])
                ce = int(indptr_arr[col_idx + 1])
                if ce <= cs:
                    continue
                rows = indices_arr[cs:ce].astype(np.int64)
                new_rows = row_map[rows]
                keep = (new_rows >= 0)
                if keep.any():
                    all_rows.append(new_rows[keep])
                    all_cols.append(np.full(int(keep.sum()), col_idx, dtype=np.int32))
                    all_vals.append(data_arr[cs:ce][keep].astype(np.float32))

                if col_idx % 10000 == 0:
                    log.info(f"    Exporting X (CSC): col {col_idx:,}/{n_cols:,}")

            if all_rows:
                cat_r = np.concatenate(all_rows)
                cat_c = np.concatenate(all_cols)
                cat_v = np.concatenate(all_vals)
                del all_rows, all_cols, all_vals
                gc.collect()

                mat_full = sp.csr_matrix((cat_v, (cat_r, cat_c)), shape=(n_kept, n_vars), dtype=np.float32)
                del cat_r, cat_c, cat_v
                gc.collect()

                for start in range(0, n_kept, chunk_size):
                    end = min(start + chunk_size, n_kept)
                    writer.append_chunk(mat_full[start:end])
                del mat_full
                gc.collect()
            else:
                for start in range(0, n_kept, chunk_size):
                    end = min(start + chunk_size, n_kept)
                    writer.append_chunk(sp.csr_matrix((end - start, n_vars), dtype=np.float32))

            writer.finalize()
        else:
            log.warning(f"Unknown sparse encoding '{enc}'. Falling back to in-memory load.")
    else:
        # Dense dataset
        for start in range(0, n_initial, chunk_size):
            end = min(start + chunk_size, n_initial)
            chunk_mask = combined_mask[start:end]
            if not chunk_mask.any():
                continue
            block = x_src[start:end]
            sub_block = block[chunk_mask]
            mat = sp.csr_matrix(sub_block, dtype=np.float32)
            writer.append_chunk(mat)
            del block, sub_block, mat
            gc.collect()
        writer.finalize()


def process_prism_dataset(
    h5ad_path: str,
    output_dir: str,
    double_pert_method: str = "both",
    overwrite: bool = False,
    dry_run: bool = False,
    output_suffix: str = "_standardized.h5ad",
) -> dict:
    """Process a single PRISM dataset through the 3-step filtering pipeline."""
    base_filename = os.path.basename(h5ad_path)
    gse_id = re.sub(r"(_standardized)?\.h5ad$", "", base_filename)
    ds_name = f"prism_{gse_id.lower()}"
    out_filename = f"{ds_name}{output_suffix}"
    out_path = os.path.join(output_dir, out_filename)

    log.info(f"\n{'='*70}")
    log.info(f"Processing PRISM Dataset: [{gse_id}] ({base_filename})")
    log.info(f"Target Output: {out_path}")
    log.info(f"{'='*70}")

    if os.path.exists(out_path) and not overwrite and not dry_run:
        log.info(f"  [*] Output file already exists: {out_path}. Skipping (use --overwrite to replace).")
        return {"dataset": gse_id, "status": "skipped", "reason": "output exists"}

    t0 = time.time()

    # Step 0: Fast metadata loading
    obs, n_vars = load_obs_fast(h5ad_path)
    n_initial = len(obs)
    log.info(f"  [Init] Total initial cells: {n_initial:,} | Genes: {n_vars:,}")

    # Step 1: CRISPRi filter
    crispri_mask, raw_crispr_types = filter_crispri_mask(obs)
    n_crispri = int(crispri_mask.sum())
    n_non_crispri = n_initial - n_crispri
    log.info(f"  [Step 1 - CRISPRi] Retained CRISPRi: {n_crispri:,} cells (Filtered out {n_non_crispri:,} non-CRISPRi)")
    log.info(f"         crispr_type breakdown: {raw_crispr_types}")

    if n_crispri == 0:
        log.warning(f"  [!] No CRISPRi cells found in {gse_id}. Skipping.")
        return {"dataset": gse_id, "status": "no_crispri_cells", "initial_cells": n_initial}

    # Slice obs to CRISPRi cells
    obs_crispri = obs[crispri_mask].copy()

    # Step 2: Control standardization (condition/targeting -> perturbation_name = 'non-targeting')
    obs_ctrl_std, n_ctrl_replaced = standardize_control_perturbations(obs_crispri)
    log.info(f"  [Step 2 - Control Std] Standardized {n_ctrl_replaced:,} cells to 'non-targeting'")

    # Step 3: Single perturbation filtering
    single_mask, pert_stats = filter_single_perturbation_mask(obs_ctrl_std, method=double_pert_method)
    n_final = int(single_mask.sum())
    log.info(f"  [Step 3 - Single Pert] Removed {pert_stats['total_multiplex_removed']:,} double/multiplex cells:")
    log.info(f"         - By guide/feature count: {pert_stats['multiplex_by_count']:,}")
    log.info(f"         - By delimiter (+, ;, ,): {pert_stats['multiplex_by_delim']:,}")
    log.info(f"         - Final retained: {n_final:,} cells ({pert_stats['single_pert_retained']:,} test + {pert_stats['control_retained']:,} control)")

    # Overall boolean mask mapping back to original dataset
    combined_mask = np.zeros(n_initial, dtype=bool)
    crispri_indices = np.where(crispri_mask)[0]
    final_indices = crispri_indices[single_mask]
    combined_mask[final_indices] = True

    retention_pct = (n_final / n_initial * 100) if n_initial > 0 else 0.0
    log.info(f"  [Summary] Final Retention: {n_final:,} / {n_initial:,} cells ({retention_pct:.2f}%)")

    if dry_run:
        log.info(f"  [DRY-RUN] Verification complete. No file written.")
        return {
            "dataset": gse_id,
            "status": "dry_run",
            "initial_cells": n_initial,
            "crispri_cells": n_crispri,
            "controls_standardized": n_ctrl_replaced,
            "multiplex_removed": pert_stats["total_multiplex_removed"],
            "final_cells": n_final,
            "retention_pct": round(retention_pct, 2),
        }

    # Step 4: Streamlined AnnData slice and export via chunked I/O
    log.info(f"  [Export] Slicing and exporting {n_final:,} cells to {out_path}...")
    adata = ad.read_h5ad(h5ad_path, backed="r")

    obs_export = obs_ctrl_std[single_mask].copy()
    obs_export = sanitize_dataframe(obs_export)
    var_export = sanitize_dataframe(adata.var.copy())

    obsm_export = {}
    if hasattr(adata, "obsm") and len(adata.obsm) > 0:
        obsm_export = {k: adata.obsm[k][combined_mask] for k in adata.obsm.keys()}

    uns_export = {}
    if hasattr(adata, "uns") and len(adata.uns) > 0:
        uns_export = dict(adata.uns)

    adata.file.close()
    del adata
    gc.collect()

    os.makedirs(output_dir, exist_ok=True)
    tmp_out = out_path + ".tmp"

    # Write skeletal AnnData structure first
    ad.AnnData(obs=obs_export, var=var_export, uns=uns_export, obsm=obsm_export).write_h5ad(tmp_out)
    gc.collect()

    # Stream X chunk by chunk directly into tmp_out
    with h5py.File(h5ad_path, "r") as h5src, h5py.File(tmp_out, "a") as h5dst:
        stream_slice_X(h5src, h5dst, combined_mask, n_vars=n_vars, chunk_size=25000)

    os.replace(tmp_out, out_path)

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f"  [Done] Saved {out_path} ({file_size_mb:.1f} MB, {elapsed:.1f}s)")

    return {
        "dataset": gse_id,
        "status": "success",
        "initial_cells": n_initial,
        "crispri_cells": n_crispri,
        "controls_standardized": n_ctrl_replaced,
        "multiplex_removed": pert_stats["total_multiplex_removed"],
        "final_cells": n_final,
        "retention_pct": round(retention_pct, 2),
        "output_path": out_path,
        "size_mb": round(file_size_mb, 1),
        "elapsed_sec": round(elapsed, 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter PRISM datasets: Extract CRISPRi cells, standardize controls to 'non-targeting', and filter out double/multiplex perturbations."
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing PRISM raw .h5ad files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Target output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--dataset",
        "-d",
        nargs="+",
        default=None,
        help="Specific GSE dataset(s) to process (e.g. GSE221321 GSE165291). Defaults to target PRISM list.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process ALL .h5ad files in the input directory.",
    )
    parser.add_argument(
        "--double-pert-method",
        choices=["both", "count", "delimiter"],
        default="both",
        help="Method to detect double/multiplex perturbations: 'both' (guide count > 1 OR delimiter '+', ';', ','), 'count', or 'delimiter' (default: both).",
    )
    parser.add_argument(
        "--suffix",
        default="_standardized.h5ad",
        help="Output filename suffix (default: '_standardized.h5ad' for downstream pipeline integration).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview cell counts, filtering breakdown, and retention ratios without writing any files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    log.info("==================================================")
    log.info("PRISM CRISPRi & Single Perturbation Filtering")
    log.info(f"Input Directory:  {args.input_dir}")
    log.info(f"Output Directory: {args.output_dir}")
    log.info(f"Double Pert Method: {args.double_pert_method}")
    log.info(f"Dry-run Mode:     {args.dry_run}")
    log.info("==================================================")

    # Determine files to process
    if args.all:
        h5ad_files = sorted(glob.glob(os.path.join(args.input_dir, "*.h5ad")))
    elif args.dataset:
        h5ad_files = []
        for d in args.dataset:
            fname = d if d.endswith(".h5ad") else f"{d}.h5ad"
            fpath = os.path.join(args.input_dir, fname)
            if os.path.exists(fpath):
                h5ad_files.append(fpath)
            else:
                log.warning(f"File not found: {fpath}")
    else:
        # Default target GSEs
        h5ad_files = []
        for gse in DEFAULT_TARGET_GSES:
            fpath = os.path.join(args.input_dir, f"{gse}.h5ad")
            if os.path.exists(fpath):
                h5ad_files.append(fpath)
            else:
                log.warning(f"Target PRISM file not found: {fpath}")

    if not h5ad_files:
        log.error("No valid .h5ad files found to process.")
        sys.exit(1)

    log.info(f"Found {len(h5ad_files)} dataset(s) to process:")
    for f in h5ad_files:
        log.info(f"  • {os.path.basename(f)}")

    results = []
    for f in h5ad_files:
        try:
            res = process_prism_dataset(
                h5ad_path=f,
                output_dir=args.output_dir,
                double_pert_method=args.double_pert_method,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                output_suffix=args.suffix,
            )
            results.append(res)
        except Exception as e:
            log.error(f"Error processing {f}: {e}", exc_info=True)
            results.append({"dataset": os.path.basename(f), "status": "error", "error": str(e)})

    # Summary table
    log.info("\n" + "=" * 80)
    log.info("FILTERING SUMMARY TABLE")
    log.info("=" * 80)
    summary_df = pd.DataFrame(results)
    cols_to_print = [c for c in ["dataset", "status", "initial_cells", "crispri_cells", "controls_standardized", "multiplex_removed", "final_cells", "retention_pct", "size_mb", "elapsed_sec"] if c in summary_df.columns]
    log.info("\n" + summary_df[cols_to_print].to_string(index=False))
    log.info("=" * 80)


if __name__ == "__main__":
    main()
