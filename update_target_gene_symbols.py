"""
update_target_gene_symbols.py
-----------------------------
Updates non-standard or Ensembl-ID formatted target gene entries in standardized h5ad files
and syncs metadata JSON files:

1. Target Gene Corrections:
   - 'ENSG00000170846' -> 'AC093323.1' (GENCODE v32 official symbol)
   - 'ENSG00000230707' -> 'AL589987.1' (GENCODE v32 official symbol)
   - 'AHSA2'           -> 'AHSA2P'     (GENCODE v32 pseudogene symbol)

2. Targets Updated:
   - obs['target_gene'] in standardized h5ad files via h5py (fast in-place category update)
   - /mnt/nas2/projects/vcc-data/standardized/metadata.json
   - /home/dev02/integration/metadata.json

Usage:
    python update_target_gene_symbols.py
    python update_target_gene_symbols.py --dry-run
"""

import argparse
import json
import logging
import os
import shutil

# Disable HDF5 file locking on NFS/NAS to prevent Errno 11
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STANDARDIZED_DIR = "/mnt/nas2/projects/vcc-data/standardized"
LOCAL_META = "/home/dev02/integration/metadata.json"
NAS_META = "/mnt/nas2/projects/vcc-data/standardized/metadata.json"

TARGET_GENE_MAPPING = {
    "ENSG00000170846": "AC093323.1",
    "ENSG00000230707": "AL589987.1",
    "AHSA2":           "AHSA2P",
}


def update_h5ad_target_genes(h5ad_path: str, mapping: dict, dry_run: bool = False) -> int:
    """Updates target_gene categories or array values in h5ad obs group."""
    ds_name = os.path.basename(h5ad_path)
    n_updated = 0

    with h5py.File(h5ad_path, "r") as f:
        if "obs" not in f or "target_gene" not in f["obs"]:
            return 0

        tg = f["obs"]["target_gene"]
        if isinstance(tg, h5py.Group) and "categories" in tg:
            cats = [c.decode() if isinstance(c, bytes) else str(c) for c in tg["categories"][:]]
            matching = [c for c in cats if c in mapping]
            if not matching:
                return 0
            n_updated = len(matching)
            log.info(f"[{ds_name}] Found {n_updated} target genes to update: {matching}")
            for old_val in matching:
                log.info(f"   - {old_val} -> {mapping[old_val]}")
        else:
            vals = set(c.decode() if isinstance(c, bytes) else str(c) for c in tg[:])
            matching = [c for c in vals if c in mapping]
            if not matching:
                return 0
            n_updated = len(matching)
            log.info(f"[{ds_name}] Found {n_updated} target genes to update: {matching}")

    if dry_run:
        log.info(f"[{ds_name}] [DRY-RUN] Skipped h5ad modification.")
        return n_updated

    # Direct in-place update of categories in h5ad (avoids 50GB file copy overhead)
    try:
        with h5py.File(h5ad_path, "a") as f:
            tg = f["obs"]["target_gene"]
            if isinstance(tg, h5py.Group) and "categories" in tg:
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in tg["categories"][:]]
                new_cats = [mapping.get(c, c) for c in cats]

                del tg["categories"]
                dt = h5py.special_dtype(vlen=str)
                tg.create_dataset("categories", data=np.array(new_cats, dtype=dt))
            else:
                vals = [c.decode() if isinstance(c, bytes) else str(c) for c in tg[:]]
                new_vals = [mapping.get(c, c) for c in vals]
                dt = tg.dtype
                del f["obs"]["target_gene"]
                f["obs"].create_dataset("target_gene", data=np.array(new_vals, dtype=object), dtype=dt)

        log.info(f"[{ds_name}] Successfully updated {n_updated} target genes in h5ad (in-place).")

    except Exception as e:
        log.error(f"[{ds_name}] Error updating h5ad: {e}")
        raise e

    return n_updated


def update_metadata_json(json_path: str, mapping: dict, dry_run: bool = False) -> int:
    """Updates unique_perturbation lists in metadata JSON."""
    if not os.path.exists(json_path):
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    total_changes = 0
    for ds_name, entry in meta.items():
        if "unique_perturbation" in entry:
            perts = entry["unique_perturbation"]
            new_perts = []
            changed_in_ds = 0
            for p in perts:
                if p in mapping:
                    new_val = mapping[p]
                    new_perts.append(new_val)
                    changed_in_ds += 1
                    total_changes += 1
                    log.info(f"  [JSON: {ds_name}] {p} -> {new_val}")
                else:
                    new_perts.append(p)

            # Deduplicate while preserving order
            entry["unique_perturbation"] = list(dict.fromkeys(new_perts))

    if total_changes == 0:
        log.info(f"[{json_path}] No changes needed.")
        return 0

    if dry_run:
        log.info(f"[{json_path}] [DRY-RUN] {total_changes} changes detected, saving skipped.")
        return total_changes

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"[{json_path}] Saved {total_changes} target gene updates.")
    return total_changes


def parse_args():
    p = argparse.ArgumentParser(description="Update target gene symbols (Ensembl IDs / pseudogene aliases) to GENCODE v32")
    p.add_argument("--standardized-dir", default=STANDARDIZED_DIR, help="Path to standardized h5ad directory")
    p.add_argument("--dry-run", action="store_true", help="Preview updates without saving changes")
    return p.parse_args()


def main():
    args = parse_args()

    log.info("=" * 65)
    log.info("Target Gene Symbol Standardization Update")
    log.info(f"Mappings to apply: {TARGET_GENE_MAPPING}")
    log.info("=" * 65)

    # 1. Update standardized h5ad files
    h5ad_files = [
        os.path.join(args.standardized_dir, f)
        for f in os.listdir(args.standardized_dir)
        if f.endswith("_standardized.h5ad")
    ]
    h5ad_files.sort()

    total_h5ad_updates = 0
    for h5ad in h5ad_files:
        cnt = update_h5ad_target_genes(h5ad, TARGET_GENE_MAPPING, dry_run=args.dry_run)
        total_h5ad_updates += cnt

    # 2. Update metadata.json files
    total_json_updates = 0
    for meta_file in [LOCAL_META, NAS_META]:
        cnt = update_metadata_json(meta_file, TARGET_GENE_MAPPING, dry_run=args.dry_run)
        total_json_updates += cnt

    log.info("\n" + "=" * 65)
    log.info("Summary of Target Gene Updates:")
    log.info(f"  H5AD files modified: {total_h5ad_updates} symbol entries updated")
    log.info(f"  JSON files modified: {total_json_updates} entries updated")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
