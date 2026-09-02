"""
standardize_perturbation_obs.py
-------------------------------
Standardizes perturbation annotations in obs across standardized h5ad files:
  1. Standardizes perturbation column name to 'target_gene'
     (nadig/replogle: 'gene', orion/kolf: 'gene_target', mixscale: 'gene',
      arc_h1: 'target_gene' (already named, content standardized), kaggle: 'sgrna_symbol')
  2. Maps target_gene values to official GENCODE v32 gene symbols (using gene_conversion / GTF)
  3. Standardizes all non-targeting labels to 'non-targeting':
     - orion_*   : CTRL -> non-targeting
     - kolf_*    : NTC  -> non-targeting
     - mixscale_*: NT   -> non-targeting
     - common: control, non_targeting, unperturbed, negctrl, ntc, nt, none, ctrl -> non-targeting

Processing Method:
  - Modifies obs in-place via h5py using temporary copy (X/layers/var untouched)
  - --dry-run: Previews mapping transformations without saving
  - --overwrite: Overwrites existing target_gene column

Usage:
    python standardize_perturbation_obs.py
    python standardize_perturbation_obs.py --dataset orion_hct116 --dry-run
    python standardize_perturbation_obs.py --overwrite
"""

import argparse
import gzip
import json
import logging
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================

META_JSON  = "/mnt/nas2/projects/vcc-data/datasets_meta.json"
GTF_GZ     = "/home/dev02/integration/gencode/gencode.v32.primary_assembly.annotation.gtf.gz"
OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"

# Canonical output column name
STD_COL = "target_gene"

# Dataset-specific source perturbation column names (mapped -> STD_COL)
DATASET_SRC_COL = {
    "nadig_hepg2":             "gene",
    "nadig_jurkat":            "gene",
    "replogle_rpe1":           "gene",
    "replogle_k562_gwp":       "gene",
    "replogle_k562_essential":  "gene",
    "orion_hct116":            "gene_target",
    "orion_hek293t":           "gene_target",
    "kolf_strong":             "gene_target",
    "mixscale_ifnb":           "gene",
    "mixscale_ifng":           "gene",
    "mixscale_ins":            "gene",
    "mixscale_tgfb":           "gene",
    "mixscale_tnfa":           "gene",
    "arc_h1":                  "target_gene",   # Already named target_gene -> standardize values
    "kaggle":                  "sgrna_symbol",
}

# Dataset-specific extra non-targeting labels (lowercase)
DATASET_NT_LABELS = {
    "orion_hct116":  {"ctrl"},
    "orion_hek293t": {"ctrl"},
    "kolf_strong":   {"ntc"},
    "mixscale_ifnb": {"nt"},
    "mixscale_ifng": {"nt"},
    "mixscale_ins":  {"nt"},
    "mixscale_tgfb": {"nt"},
    "mixscale_tnfa": {"nt"},
}

# Common non-targeting labels (lowercase)
COMMON_NT_LABELS = {
    "non-targeting", "non_targeting", "control", "unperturbed",
    "nan", "negctrl", "ntc", "nt", "none", "ctrl", "negative_control",
    "non targeting",
}

STANDARD_NT = "non-targeting"

# Specific target gene symbol overrides to official GENCODE v32 symbols
TARGET_GENE_OVERRIDE = {
    "ENSG00000170846": "AC093323.1",
    "ENSG00000230707": "AL589987.1",
    "AHSA2":           "AHSA2P",
}


# ============================================================
# 1. Load GENCODE v32
# ============================================================

def load_gencode_v32(gtf_gz: str) -> tuple:
    """Returns Ensembl ID -> gene symbol mapping and v32 gene symbol set."""
    log.info(f"Loading GENCODE v32 GTF: {gtf_gz}")
    ens2sym = {}
    v32_set = set()
    with gzip.open(gtf_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.strip().split("\t")
            if len(fields) < 9 or fields[2] != "gene":
                continue
            attrs = fields[8]
            gene_name = gene_id = None
            for item in attrs.split(";"):
                item = item.strip()
                if item.startswith("gene_name"):
                    gene_name = item.split('"')[1] if '"' in item else item.split(" ", 1)[1]
                elif item.startswith("gene_id"):
                    gene_id = item.split('"')[1] if '"' in item else item.split(" ", 1)[1]
            if gene_name:
                v32_set.add(gene_name)
            if gene_id and gene_name:
                ens2sym[gene_id] = gene_name
                ens2sym[gene_id.split(".")[0]] = gene_name
    log.info(f"  GENCODE v32: {len(v32_set):,} symbols, {len(ens2sym):,} Ensembl IDs")
    return ens2sym, v32_set


# ============================================================
# 2. Value Conversion Function
# ============================================================

def apply_conversion(val: str, ds_name: str, conv: dict,
                     ens2sym: dict, v32_set: set) -> str:
    """Standardizes a single perturbation value."""
    v = str(val).strip()
    v_lower = v.lower()

    # Check non-targeting
    nt_labels = COMMON_NT_LABELS | DATASET_NT_LABELS.get(ds_name, set())
    if not v or v_lower in nt_labels:
        return STANDARD_NT

    # Strip guide suffix (e.g., TP53_1 -> TP53)
    base = v.rsplit("_", 1)[0] if "_" in v and v.rsplit("_", 1)[1].isdigit() else v
    if base.lower() in nt_labels:
        return STANDARD_NT

    # Direct target gene override (e.g. Ensembl ID to v32 symbol, pseudogene alias)
    if v in TARGET_GENE_OVERRIDE:
        return TARGET_GENE_OVERRIDE[v]
    if base in TARGET_GENE_OVERRIDE:
        return TARGET_GENE_OVERRIDE[base]

    # Apply gene_conversion
    if base in conv:
        return conv[base]
    if v in conv:
        return conv[v]

    # Ensembl ID -> v32 symbol
    if base in ens2sym:
        return ens2sym[base]
    if v in ens2sym:
        return ens2sym[v]

    # Already valid v32 symbol
    if base in v32_set:
        return base

    return base


def make_converter_series(raw_series: pd.Series, ds_name: str, conv: dict,
                          ens2sym: dict, v32_set: set) -> pd.Series:
    """Applies conversion across Series by mapping unique values for high performance."""
    unique_vals = raw_series.unique()
    mapping = {v: apply_conversion(v, ds_name, conv, ens2sym, v32_set)
               for v in unique_vals}
    return raw_series.map(mapping)


# ============================================================
# 3. Rename obs Column & Standardize Values via h5py
# ============================================================

def write_categorical_col(obs_grp: h5py.Group, col_name: str, std_series: pd.Series):
    """Write categorical column to h5py obs Group (AnnData 0.2.0 encoding)."""
    if col_name in obs_grp:
        del obs_grp[col_name]

    unique_cats = sorted(std_series.unique().tolist())
    cat_to_idx  = {c: i for i, c in enumerate(unique_cats)}

    n_cats = len(unique_cats)
    codes  = std_series.map(cat_to_idx).values
    codes  = codes.astype(np.int8 if n_cats <= 127 else np.int16)

    cat_grp = obs_grp.create_group(col_name)
    cat_grp.attrs["encoding-type"]    = "categorical"
    cat_grp.attrs["encoding-version"] = "0.2.0"
    cat_grp.attrs["ordered"]          = False

    dt = h5py.special_dtype(vlen=str)
    cat_grp.create_dataset("categories", data=np.array(unique_cats, dtype=dt))
    cat_grp.create_dataset("codes", data=codes)


def update_column_order(obs_grp: h5py.Group, old_col: str, new_col: str):
    """Update AnnData column-order attribute (old_col -> new_col)."""
    if "column-order" not in obs_grp.attrs:
        return
    col_order = list(obs_grp.attrs["column-order"])
    if old_col in col_order and new_col not in col_order:
        idx = col_order.index(old_col)
        col_order[idx] = new_col
        obs_grp.attrs["column-order"] = col_order
    elif new_col not in col_order:
        col_order.append(new_col)
        obs_grp.attrs["column-order"] = col_order


# ============================================================
# 4. Single Dataset Processing
# ============================================================

def process_dataset(ds_name: str, entry: dict,
                    ens2sym: dict, v32_set: set,
                    output_dir: str,
                    dry_run: bool = False,
                    overwrite: bool = False) -> dict:
    h5ad_path = os.path.join(output_dir, f"{ds_name}_standardized.h5ad")
    src_col   = DATASET_SRC_COL.get(ds_name)

    if not os.path.exists(h5ad_path):
        log.warning(f"[{ds_name}] File not found: {h5ad_path}")
        return {"dataset": ds_name, "status": "missing"}
    if src_col is None:
        log.warning(f"[{ds_name}] Undefined DATASET_SRC_COL, skipping")
        return {"dataset": ds_name, "status": "no_col_def"}

    log.info(f"\n[{ds_name}] Processing...")
    log.info(f"  File: {h5ad_path}")
    log.info(f"  src_col='{src_col}' -> STD_COL='{STD_COL}'")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs   = adata.obs.copy()
    n_cells = len(obs)
    adata.file.close()

    if src_col not in obs.columns:
        log.warning(f"  [{ds_name}] Column '{src_col}' not found! Available: {list(obs.columns)}")
        return {"dataset": ds_name, "status": "error", "reason": f"column '{src_col}' not found"}

    if STD_COL in obs.columns and src_col != STD_COL and not overwrite:
        log.info(f"  [{ds_name}] '{STD_COL}' already exists. Use --overwrite to force replace.")
        return {"dataset": ds_name, "status": "skipped", "reason": "target_gene column already exists"}

    # Perform conversion
    conv      = entry.get("gene_conversion", {})
    raw_vals  = obs[src_col].astype(str).fillna("nan")
    std_vals  = make_converter_series(raw_vals, ds_name, conv, ens2sym, v32_set)

    n_nt      = int((std_vals == STANDARD_NT).sum())
    n_changed = int((raw_vals != std_vals).sum())
    n_unique  = int(std_vals.nunique())

    log.info(f"  cells={n_cells:,}, NT={n_nt:,}, converted={n_changed:,}, unique={n_unique:,}")

    changed   = pd.DataFrame({"raw": raw_vals, "std": std_vals})
    changed   = changed[changed.raw != changed.std].drop_duplicates()
    if not changed.empty:
        log.info(f"  Conversion sample:\n{changed.head(10).to_string(index=False)}")

    if v32_set:
        non_nt_vals   = std_vals[std_vals != STANDARD_NT]
        unmapped      = set(non_nt_vals.unique()) - v32_set
        if unmapped:
            log.warning(f"  {len(unmapped)} genes not in GENCODE v32: {sorted(unmapped)[:10]}")

    if dry_run:
        log.info("  [DRY-RUN] Save skipped")
        return {
            "dataset": ds_name, "status": "dry_run",
            "n_cells": n_cells, "n_nt": n_nt,
            "n_changed": n_changed, "n_unique": n_unique,
        }

    tmp_path = h5ad_path + ".perttmp"
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as h5f:
            obs_grp = h5f["obs"]
            write_categorical_col(obs_grp, STD_COL, std_vals)
            update_column_order(obs_grp, src_col, STD_COL if src_col == STD_COL else STD_COL)

        shutil.move(tmp_path, h5ad_path)
        log.info(f"  [{ds_name}] Saved successfully -> {h5ad_path}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {
        "dataset": ds_name, "status": "done",
        "n_cells": n_cells, "n_nt": n_nt,
        "n_changed": n_changed, "n_unique": n_unique,
    }


# ============================================================
# 5. Main CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Standardize perturbation column in obs to target_gene")
    p.add_argument("--meta-json",  default=META_JSON, help="Path to metadata JSON")
    p.add_argument("--gtf-gz",     default=GTF_GZ, help="Path to GENCODE v32 GTF")
    p.add_argument("--output-dir", default=OUTPUT_DIR, help="Standardized h5ad directory path")
    p.add_argument("--dataset",    default=None, help="Process a specific dataset only")
    p.add_argument("--dry-run",    action="store_true", help="Preview conversion results without saving")
    p.add_argument("--overwrite",  action="store_true", help="Overwrite existing target_gene column")
    p.add_argument("--skip-gtf",   action="store_true", help="Skip GTF loading (use gene_conversion only)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.meta_json) as f:
        meta = json.load(f)

    if args.skip_gtf:
        ens2sym, v32_set = {}, set()
        log.info("GTF loading skipped (--skip-gtf)")
    else:
        ens2sym, v32_set = load_gencode_v32(args.gtf_gz)

    targets = {args.dataset: meta[args.dataset]} if args.dataset else meta

    results = []
    for ds_name, entry in targets.items():
        try:
            r = process_dataset(
                ds_name, entry, ens2sym, v32_set,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
            results.append(r)
        except Exception as e:
            log.error(f"[{ds_name}] ERROR: {e}", exc_info=True)
            results.append({"dataset": ds_name, "status": "error", "reason": str(e)})

    # Summary table
    log.info("\n" + "=" * 65)
    log.info("Results Summary:")
    log.info(f"{'Dataset':<35} {'Status':<12} {'Cells':>8} {'NT':>8} {'Converted':>10} {'Unique':>8}")
    log.info("-" * 85)
    for r in results:
        s = r.get("status", "?")
        if s in ("done", "dry_run"):
            log.info(
                f"  {r['dataset']:<35} {s:<12} {r.get('n_cells', 0):>8,} "
                f"{r.get('n_nt', 0):>8,} {r.get('n_changed', 0):>10,} "
                f"{r.get('n_unique', 0):>8,}"
            )
        else:
            log.info(f"  {r['dataset']:<35} {s:<12}  {r.get('reason', '')}")

    n_ok = sum(1 for r in results if r.get("status") in ("done", "dry_run", "skipped"))
    log.info(f"\nProcessing completed: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
