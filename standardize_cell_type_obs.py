"""
standardize_cell_type_obs.py
----------------------------
Standardizes or injects the 'cell_type' column in obs across standardized h5ad files or arbitrary h5ad files.

Key Features:
  1. Missing cell_type: Generates a single-category categorical column with the specified cell_type name (e.g., 'hct116')
  2. Legacy celltype column (e.g., kolf_strong):
     - If --cell-type is provided, assigns that value to 'cell_type'
     - If not provided, copies/renames the existing 'celltype' values to 'cell_type'
  3. Existing cell_type (e.g., mixscale_*, arc_h1):
     - Skips unless --overwrite is explicitly provided (safety mechanism)

Processing & Safety:
  - By default, leaves the original raw input file UNTOUCHED and writes the standardized result to --output-file or --output-dir (e.g. /mnt/nas2/projects/vcc-data/standardized/<name>_standardized.h5ad).
  - Use --inplace only if you explicitly want to modify the input file directly.
  - Writes AnnData Categorical specification (categories + codes) directly via h5py (RAM/disk efficient)

Usage:
    # Process raw h5ad and save to standardized directory (Original file untouched):
    python standardize_cell_type_obs.py --h5ad /mnt/nas2/projects/vcc-data/vcc-2026/context_A.h5ad --cell-type CONTEXT_A

    # Process raw h5ad and save to a specific output file:
    python standardize_cell_type_obs.py --h5ad /path/to/sample.h5ad --output-file /path/to/sample_standardized.h5ad --cell-type hepg2

    # In-place modification (Modifies input file directly):
    python standardize_cell_type_obs.py --h5ad /path/to/sample.h5ad --cell-type hepg2 --inplace --overwrite

    # Dataset name lookup in standardized directory:
    python standardize_cell_type_obs.py --dataset orion_hct116 --cell-type hct116

    # Batch apply to standardized directory:
    python standardize_cell_type_obs.py --all
    python standardize_cell_type_obs.py --all --dry-run
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import glob
import logging
import shutil

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"
TARGET_COL = "cell_type"

# Recommended cell_type mapping presets (used with --all)
DEFAULT_CELL_TYPES = {
    "nadig_hepg2":             "hepg2",
    "nadig_jurkat":            "jurkat",
    "replogle_rpe1":           "rpe1",
    "replogle_k562_gwp":       "k562",
    "replogle_k562_essential":  "k562",
    "orion_hct116":            "hct116",
    "orion_hek293t":           "hek293t",
    "kolf_strong":             "kolf2.1j",  # Original celltype: 'KRAB-ZIM3-dCas9 KOLF2.1J hiPSC'
    # arc_h1: already has cell_type=['ARC_H1']
    # mixscale_*: already has cell_type=['A549', 'BXPC3', 'HAP1']
    # kaggle: specify as needed
}


def get_n_cells(h5f: h5py.File) -> int:
    """Determine n_cells accurately from X or obs group."""
    # 1. Prefer X shape if available
    if "X" in h5f:
        X = h5f["X"]
        if isinstance(X, h5py.Group):
            if "shape" in X.attrs:
                return int(X.attrs["shape"][0])
            if "indptr" in X:
                return len(X["indptr"]) - 1
        elif hasattr(X, "shape") and len(X.shape) > 0:
            return int(X.shape[0])

    # 2. Check obs index
    obs = h5f.get("obs")
    if obs is not None:
        idx_key = obs.attrs.get("_index", "_index")
        if idx_key in obs:
            idx_item = obs[idx_key]
            if isinstance(idx_item, h5py.Group):
                if "codes" in idx_item:
                    return len(idx_item["codes"])
                if "values" in idx_item:
                    return len(idx_item["values"])
            elif hasattr(idx_item, "shape") and len(idx_item.shape) > 0:
                return int(idx_item.shape[0])

        # 3. Check any obs column
        for k in obs.keys():
            if k in ("__categories", "cell_type"):
                continue
            item = obs[k]
            if isinstance(item, h5py.Group):
                if "codes" in item:
                    return len(item["codes"])
                if "values" in item:
                    return len(item["values"])
            elif hasattr(item, "shape") and len(item.shape) > 0:
                return int(item.shape[0])

    raise ValueError("Unable to determine n_cells.")


def update_column_order(obs_grp: h5py.Group, col_name: str, old_col: str = None):
    """Update AnnData obs column-order attribute."""
    if "column-order" not in obs_grp.attrs:
        return
    col_order = list(obs_grp.attrs["column-order"])
    if old_col and old_col in col_order and col_name not in col_order:
        idx = col_order.index(old_col)
        col_order[idx] = col_name
    elif col_name not in col_order:
        col_order.append(col_name)
    obs_grp.attrs["column-order"] = col_order


def process_h5ad(input_path: str, output_path: str, ds_name: str,
                 cell_type_val: str = None,
                 rename_only: bool = False, overwrite: bool = False,
                 dry_run: bool = False) -> dict:
    log.info(f"\n[{ds_name}] Inspecting: {input_path}")
    if input_path != output_path:
        log.info(f"  Target output: {output_path}")

    if not os.path.exists(input_path):
        log.warning(f"  [{ds_name}] Input file does not exist: {input_path}")
        return {"dataset": ds_name, "status": "missing"}

    if os.path.exists(output_path) and input_path != output_path and not overwrite:
        log.info(f"  [{ds_name}] Output file already exists: {output_path}. Use --overwrite to replace.")
        return {"dataset": ds_name, "status": "skipped", "reason": "output file exists"}

    if os.path.exists(output_path + ".perttmp"):
        log.warning(f"  [{ds_name}] Another active job (.perttmp) detected. Skipping.")
        return {"dataset": ds_name, "status": "skipped", "reason": ".perttmp exists"}

    with h5py.File(input_path, "r") as h5f:
        obs = h5f["obs"]
        obs_keys = list(obs.keys())
        n_cells = get_n_cells(h5f)

        has_cell_type = TARGET_COL in obs_keys
        has_celltype = "celltype" in obs_keys

        # Current status description
        existing_info = []
        if has_cell_type:
            ds = obs[TARGET_COL]
            if isinstance(ds, h5py.Group) and "categories" in ds:
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][:5]]
                existing_info.append(f"cell_type categories={cats}")
            else:
                existing_info.append("cell_type (array)")
        if has_celltype:
            ds = obs["celltype"]
            if isinstance(ds, h5py.Group) and "categories" in ds:
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][:5]]
                existing_info.append(f"celltype categories={cats}")
            else:
                existing_info.append("celltype (array)")

        log.info(f"  cells: {n_cells:,}, existing columns: {existing_info if existing_info else 'none'}")

        # Decide processing action
        action = None
        target_value_desc = ""

        if has_cell_type and input_path == output_path and not overwrite:
            log.info(f"  [{ds_name}] Column '{TARGET_COL}' already exists in input file. Use --overwrite to modify.")
            return {"dataset": ds_name, "status": "skipped", "reason": "already has cell_type"}

        if rename_only:
            if not has_celltype:
                log.warning(f"  [{ds_name}] --rename-only specified but no 'celltype' column found.")
                return {"dataset": ds_name, "status": "skipped", "reason": "no celltype column to rename"}
            action = "rename_celltype"
            target_value_desc = "preserve celltype values and rename"
        elif cell_type_val:
            action = "assign_new"
            target_value_desc = f"assign '{cell_type_val}'"
        elif has_celltype:
            action = "rename_celltype"
            target_value_desc = "preserve celltype values and copy"
        else:
            log.warning(f"  [{ds_name}] No --cell-type value specified and no existing 'celltype' column found.")
            return {"dataset": ds_name, "status": "skipped", "reason": "no cell_type value given"}

    log.info(f"  Action: {action} ({target_value_desc})")

    if dry_run:
        log.info(f"  [DRY-RUN] File modification skipped.")
        return {"dataset": ds_name, "status": "dry_run", "action": action, "value": target_value_desc}

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Copy input to temporary output file
    tmp_path = output_path + ".ct_tmp"
    shutil.copyfile(input_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as h5f:
            obs = h5f["obs"]

            if action == "assign_new":
                if TARGET_COL in obs:
                    del obs[TARGET_COL]

                # Create as single-category categorical (saves massive RAM/disk)
                cat_grp = obs.create_group(TARGET_COL)
                cat_grp.attrs["encoding-type"]    = "categorical"
                cat_grp.attrs["encoding-version"] = "0.2.0"
                cat_grp.attrs["ordered"]          = False

                dt = h5py.string_dtype(encoding="utf-8")
                cat_grp.create_dataset("categories", data=np.array([cell_type_val], dtype=object), dtype=dt)
                codes = np.zeros(n_cells, dtype=np.int8)
                cat_grp.create_dataset("codes", data=codes)

                update_column_order(obs, TARGET_COL, old_col="celltype" if has_celltype else None)

            elif action == "rename_celltype":
                src_ds = obs["celltype"]
                if TARGET_COL in obs:
                    del obs[TARGET_COL]

                if isinstance(src_ds, h5py.Group) and "categories" in src_ds:
                    cat_grp = obs.create_group(TARGET_COL)
                    for k, v in src_ds.attrs.items():
                        cat_grp.attrs[k] = v
                    cat_grp.attrs["encoding-type"]    = "categorical"
                    cat_grp.attrs["encoding-version"] = "0.2.0"
                    cat_grp.attrs["ordered"]          = False

                    cat_grp.create_dataset("categories", data=src_ds["categories"][:], dtype=src_ds["categories"].dtype)
                    cat_grp.create_dataset("codes", data=src_ds["codes"][:], dtype=src_ds["codes"].dtype)
                else:
                    obs.create_dataset(TARGET_COL, data=src_ds[:], dtype=src_ds.dtype)
                    obs[TARGET_COL].attrs["encoding-type"] = "string-array"
                    obs[TARGET_COL].attrs["encoding-version"] = "0.2.0"

                update_column_order(obs, TARGET_COL, old_col="celltype")

        shutil.move(tmp_path, output_path)
        log.info(f"  [{ds_name}] Saved successfully -> {output_path}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {"dataset": ds_name, "status": "done", "action": action, "value": target_value_desc}


def parse_args():
    p = argparse.ArgumentParser(description="Standardize and inject cell_type in AnnData obs")
    p.add_argument("--h5ad", default=None, help="Direct path to an input h5ad file")
    p.add_argument("--output-file", default=None, help="Explicit destination file path (leaves original input untouched)")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Standardized output directory path (default: /mnt/nas2/projects/vcc-data/standardized)")
    p.add_argument("--inplace", action="store_true", help="Modify the input file directly in-place")
    p.add_argument("--dataset", default=None, help="Target dataset name (e.g., orion_hct116, nadig_hepg2) or path")
    p.add_argument("--cell-type", default=None, help="cell_type value to inject (e.g., hct116, hepg2)")
    p.add_argument("--rename-only", action="store_true", help="Only copy/rename existing 'celltype' to 'cell_type'")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file or existing column")
    p.add_argument("--all", action="store_true", help="Batch process all datasets in output-dir using presets")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.all and not args.dataset and not args.h5ad:
        log.error("Error: Specify --h5ad [path], --dataset [name], or --all. (See --help)")
        return

    results = []

    # Case 1: Direct h5ad path via --h5ad or --dataset pointing to an existing file
    target_h5ad = args.h5ad
    if not target_h5ad and args.dataset and os.path.exists(args.dataset):
        target_h5ad = args.dataset

    if target_h5ad:
        if not os.path.exists(target_h5ad):
            log.error(f"Error: File not found: {target_h5ad}")
            return
        ds_name = args.dataset if (args.dataset and not os.path.exists(args.dataset)) else os.path.splitext(os.path.basename(target_h5ad))[0]

        if args.output_file:
            out_path = args.output_file
        elif args.inplace:
            out_path = target_h5ad
        else:
            clean_name = ds_name.replace("_standardized", "")
            out_path = os.path.join(args.output_dir, f"{clean_name}_standardized.h5ad")

        r = process_h5ad(target_h5ad, out_path, ds_name, cell_type_val=args.cell_type,
                         rename_only=args.rename_only, overwrite=args.overwrite,
                         dry_run=args.dry_run)
        results.append(r)

    elif args.all:
        log.info("Batch processing all datasets in output-dir using default presets (--all)")
        files = sorted(glob.glob(os.path.join(args.output_dir, "*_standardized.h5ad")))
        for f in files:
            ds_name = os.path.basename(f).replace("_standardized.h5ad", "")
            preset_val = DEFAULT_CELL_TYPES.get(ds_name, None)

            if preset_val is None:
                if ds_name == "kolf_strong":
                    r = process_h5ad(f, f, ds_name, cell_type_val=None, rename_only=True,
                                     overwrite=args.overwrite, dry_run=args.dry_run)
                else:
                    log.info(f"[{ds_name}] No preset or skipped")
                    r = {"dataset": ds_name, "status": "skipped", "reason": "no preset"}
            else:
                r = process_h5ad(f, f, ds_name, cell_type_val=preset_val, rename_only=False,
                                 overwrite=args.overwrite, dry_run=args.dry_run)
            results.append(r)
    else:
        clean_name = args.dataset.replace("_standardized.h5ad", "").replace(".h5ad", "")
        h5ad_path = os.path.join(args.output_dir, f"{clean_name}_standardized.h5ad")
        if not os.path.exists(h5ad_path):
            alt_path = os.path.join(args.output_dir, f"{clean_name}.h5ad")
            if os.path.exists(alt_path):
                h5ad_path = alt_path

        out_path = args.output_file if args.output_file else h5ad_path

        r = process_h5ad(h5ad_path, out_path, clean_name, cell_type_val=args.cell_type,
                         rename_only=args.rename_only, overwrite=args.overwrite,
                         dry_run=args.dry_run)
        results.append(r)

    # Summary table
    log.info("\n" + "=" * 65)
    log.info("Results Summary:")
    log.info(f"{'Dataset':<25} {'Status':<12} {'Action/Details'}")
    log.info("-" * 65)
    for r in results:
        status = r.get("status", "?")
        desc = r.get("value") or r.get("reason") or ""
        log.info(f"  {r['dataset']:<23} {status:<12} {desc}")


if __name__ == "__main__":
    main()
