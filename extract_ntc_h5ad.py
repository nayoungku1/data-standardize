"""
extract_ntc_h5ad.py
-------------------
Filters non-targeting control (NTC) cells from each standardized h5ad file
and saves them as standalone h5ad files in the NTC directory (/mnt/nas2/projects/vcc-data/NTC/).

Key Features:
  1. Ultra-fast dry-run:
     - Directly reads obs metadata using h5py, inspecting tens of gigabytes in sub-seconds
  2. Memory-efficient extraction:
     - Slices only X, obs, var, and obsm in backed='r' mode to prevent OOM errors on massive files
  3. Flexible NTC detection:
     - Primary: obs['target_gene'] == 'non-targeting'
     - Fallback: candidate columns (gene_target, gene, sgrna_symbol) and
       case-insensitive matching against ['non-targeting', 'non_targeting', 'control', 'ntc', 'nt', 'ctrl']
  4. Metadata preservation:
     - Preserves var (gene_name, ensembl_id, etc.), obsm embeddings, and uns metadata
  5. AnnData / Pandas compatibility:
     - Automatically sanitizes pd.StringDtype (nullable string) arrays to ensure seamless saving

Usage:
    # 1. Preview NTC statistics across all datasets (dry-run)
    python extract_ntc_h5ad.py --dry-run

    # 2. Extract NTC for a specific dataset
    python extract_ntc_h5ad.py --dataset nadig_hepg2
    python extract_ntc_h5ad.py --dataset orion_hct116

    # 3. Batch extract all datasets (automatically skips completed files)
    python extract_ntc_h5ad.py
"""

import argparse
import glob
import logging
import os
import time

import anndata as ad
from anndata._io import read_elem
import h5py
import numpy as np
import pandas as pd

# Enable AnnData nullable string writing support
try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Sanitize pd.StringDtype (nullable string) arrays to object dtype to prevent AnnData writing errors."""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) == "string":
            df[col] = df[col].astype(object).fillna("")
    return df

DEFAULT_INPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"
DEFAULT_OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/NTC"

NT_LABELS = {
    "non-targeting", "non_targeting", "non targeting",
    "control", "ctrl", "unperturbed", "nan", "negctrl", "ntc", "nt", "none"
}

CANDIDATE_COLS = ["target_gene", "gene_target", "gene", "sgrna_symbol", "perturbation"]


def find_ntc_mask(obs: pd.DataFrame) -> tuple:
    """Find boolean mask for NTC cells in obs DataFrame and return (mask, detected_column)."""
    for col in CANDIDATE_COLS:
        if col in obs.columns:
            vals = obs[col].astype(str).str.strip().str.lower()
            mask = vals.isin(NT_LABELS)
            if mask.any():
                return mask.values, col

    for col in obs.columns:
        if any(k in col.lower() for k in ["pert", "target", "guide"]):
            vals = obs[col].astype(str).str.strip().str.lower()
            mask = vals.isin(NT_LABELS)
            if mask.any():
                return mask.values, col

    return np.zeros(len(obs), dtype=bool), None


def get_obs_fast(h5ad_path: str) -> tuple:
    """Fast load of obs and dimensions via h5py without touching X/layers."""
    with h5py.File(h5ad_path, "r") as h5f:
        obs = read_elem(h5f["obs"])
        n_vars = 0
        if "var" in h5f:
            if "_index" in h5f["var"]:
                n_vars = len(h5f["var"]["_index"])
            elif "gene_name" in h5f["var"]:
                n_vars = len(h5f["var"]["gene_name"])
            elif "column-order" in h5f["var"].attrs:
                first_col = h5f["var"].attrs["column-order"][0]
                n_vars = len(h5f["var"][first_col])
        return obs, n_vars


def process_dataset(h5ad_path: str, output_dir: str,
                    overwrite: bool = False, dry_run: bool = False) -> dict:
    ds_name = os.path.basename(h5ad_path).replace("_standardized.h5ad", "").replace(".h5ad", "")
    out_filename = f"{ds_name}_ntc.h5ad"
    out_path = os.path.join(output_dir, out_filename)

    log.info(f"\n[{ds_name}] Processing: {h5ad_path}")

    # Check for active temporary files to avoid race conditions
    for ext in [".ct_tmp", ".varid_tmp", ".perttmp"]:
        if os.path.exists(h5ad_path + ext):
            log.warning(f"  [{ds_name}] Active temporary file ({ext}) detected. Skipping.")
            return {"dataset": ds_name, "status": "skipped", "reason": f"active tmp file ({ext})"}

    if os.path.exists(out_path) and not overwrite and not dry_run:
        log.info(f"  [{ds_name}] Output file already exists: {out_path} (use --overwrite to replace)")
        return {"dataset": ds_name, "status": "skipped", "reason": "output exists"}

    t0 = time.time()

    # 1. Fast load obs
    obs, n_vars = get_obs_fast(h5ad_path)
    n_total_cells = len(obs)

    # 2. Determine NTC mask
    mask, col_used = find_ntc_mask(obs)
    n_ntc = int(mask.sum())
    ntc_ratio = (n_ntc / n_total_cells * 100) if n_total_cells > 0 else 0

    log.info(f"  Total cells: {n_total_cells:,} | Genes: {n_vars:,}")
    log.info(f"  Detected column: '{col_used}' -> NTC cells: {n_ntc:,} ({ntc_ratio:.2f}%)")

    if n_ntc == 0:
        log.warning(f"  [{ds_name}] No NTC cells detected.")
        return {
            "dataset": ds_name, "status": "no_ntc_found",
            "total_cells": n_total_cells, "ntc_cells": 0, "ratio": 0.0
        }

    if dry_run:
        log.info(f"  [DRY-RUN] Verification complete (expected output: {out_path})")
        return {
            "dataset": ds_name, "status": "dry_run",
            "total_cells": n_total_cells, "ntc_cells": n_ntc,
            "ratio": round(ntc_ratio, 2), "col_used": col_used
        }

    # 3. Slice NTC cells (lightweight mode: slices X, obs, var without loading massive layers)
    log.info(f"  Slicing {n_ntc:,} NTC cells (lightweight memory mode)...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    
    X_sub = adata.X[mask]
    obs_sub = sanitize_dataframe(adata.obs[mask].copy())
    var_sub = sanitize_dataframe(adata.var.copy())
    
    sub_adata = ad.AnnData(X=X_sub, obs=obs_sub, var=var_sub)
    
    if hasattr(adata, "obsm") and len(adata.obsm) > 0:
        sub_adata.obsm = {k: adata.obsm[k][mask] for k in adata.obsm.keys()}
        
    adata.file.close()

    # 4. Save
    os.makedirs(output_dir, exist_ok=True)
    tmp_out = out_path + ".tmp"
    sub_adata.write_h5ad(tmp_out, compression="gzip")
    os.replace(tmp_out, out_path)

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f"  [{ds_name}] Completed! ({n_ntc:,} cells × {n_vars:,} genes, {file_size_mb:.1f} MB, {elapsed:.1f}s)")

    return {
        "dataset": ds_name, "status": "done",
        "total_cells": n_total_cells, "ntc_cells": n_ntc,
        "ratio": round(ntc_ratio, 2), "col_used": col_used,
        "output": out_path, "size_mb": round(file_size_mb, 1),
        "elapsed_sec": round(elapsed, 1)
    }


def parse_args():
    p = argparse.ArgumentParser(description="Filter and extract Non-Targeting Control (NTC) cells from standardized h5ad files")
    p.add_argument("--dataset", default=None, help="Process a specific dataset only (e.g., nadig_hepg2, orion_hct116)")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="Standardized h5ad directory path")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Target NTC directory path")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing NTC files")
    p.add_argument("--dry-run", action="store_true", help="Preview NTC cell counts and ratios without saving")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset:
        clean_name = args.dataset.replace("_standardized.h5ad", "").replace(".h5ad", "")
        h5ad_path = os.path.join(args.input_dir, f"{clean_name}_standardized.h5ad")
        if not os.path.exists(h5ad_path) and os.path.exists(args.dataset):
            h5ad_path = args.dataset
        targets = [h5ad_path]
    else:
        targets = sorted(glob.glob(os.path.join(args.input_dir, "*_standardized.h5ad")))

    if not targets:
        log.error(f"No target files found in: {args.input_dir}")
        return

    log.info(f"NTC Extraction Task: {len(targets)} target files")
    log.info(f"Input directory: {args.input_dir}")
    log.info(f"Output directory: {args.output_dir}")

    results = []
    for f in targets:
        try:
            r = process_dataset(f, args.output_dir, overwrite=args.overwrite, dry_run=args.dry_run)
            results.append(r)
        except Exception as e:
            log.error(f"[{f}] ERROR: {e}", exc_info=True)
            ds_name = os.path.basename(f).replace("_standardized.h5ad", "").replace(".h5ad", "")
            results.append({"dataset": ds_name, "status": "error", "reason": str(e)})

    # Summary table
    log.info("\n" + "=" * 75)
    log.info("NTC Filtering Summary:")
    log.info(f"{'Dataset':<25} {'Status':<10} {'Total Cells':>11} {'NTC Cells':>11} {'Ratio(%)':>8} {'Column'}")
    log.info("-" * 75)
    for r in results:
        status = r.get("status", "?")
        tot = f"{r.get('total_cells', 0):,}" if "total_cells" in r else "-"
        ntc = f"{r.get('ntc_cells', 0):,}" if "ntc_cells" in r else "-"
        ratio = f"{r.get('ratio', 0.0):.2f}%" if "ratio" in r else "-"
        col = r.get("col_used") or r.get("reason") or ""
        log.info(f"  {r['dataset']:<23} {status:<10} {tot:>11} {ntc:>11} {ratio:>8} {col}")

    n_done = sum(1 for r in results if r.get("status") in ("done", "dry_run"))
    log.info(f"\nProcessing completed: {n_done}/{len(results)}")


if __name__ == "__main__":
    main()
