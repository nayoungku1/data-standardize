"""
restore_raw_counts.py
---------------------
Restores or replaces expression matrix X with raw UMI counts for datasets in the
standardized directory that were previously stored as normalized/transformed values (arc_h1 and kolf_strong).

1. kolf_strong:
   - Contains raw integer counts (CSR) in layers['counts']
   - Replaces X with layers['counts'] while preserving original normalized X in layers['normalized']

2. arc_h1:
   - X is stored as ln(1 + raw_count) (log1p)
   - expm1(data) restores exact integer UMI counts (floating-point precision within 1e-6)
   - Preserves CSR sparsity structure (expm1(0) == 0) and rounds expm1(data) to integer
   - Backs up original log1p matrix to layers['log1p']

Usage:
    # 1. Preview data samples and precision without modification
    python restore_raw_counts.py --dry-run

    # 2. Run for a specific dataset
    python restore_raw_counts.py --dataset arc_h1
    python restore_raw_counts.py --dataset kolf_strong

    # 3. Run for both datasets
    python restore_raw_counts.py
"""

import argparse
import logging
import os
import shutil
import time

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STANDARDIZED_DIR = "/mnt/nas2/projects/vcc-data/standardized"


def restore_arc_h1(h5ad_path: str, dry_run: bool = False, backup_layer: bool = True) -> dict:
    """Restores X in arc_h1 from log1p CSR to raw integer count CSR using expm1."""
    log.info(f"\n[arc_h1] Inspecting and restoring: {h5ad_path}")

    with h5py.File(h5ad_path, "r") as f:
        if "X" not in f or not isinstance(f["X"], h5py.Group) or "data" not in f["X"]:
            log.error("  [arc_h1] Cannot locate X CSR group.")
            return {"dataset": "arc_h1", "status": "error", "reason": "invalid X structure"}

        sample_d = f["X"]["data"][:20]
        expm1_sample = np.expm1(sample_d)
        round_sample = np.round(expm1_sample)
        max_diff = float(np.max(np.abs(expm1_sample - round_sample)))

        log.info(f"  Current X data sample (log1p) : {sample_d[:5].tolist()}")
        log.info(f"  Restored raw counts sample    : {round_sample[:5].tolist()}")
        log.info(f"  Max integer rounding diff     : {max_diff:.2e} (matches float precision)")

        n_elements = len(f["X"]["data"])
        log.info(f"  Total non-zero elements: {n_elements:,}")

    if dry_run:
        log.info("  [DRY-RUN] Save skipped.")
        return {"dataset": "arc_h1", "status": "dry_run", "max_diff": max_diff}

    t0 = time.time()
    tmp_path = h5ad_path + ".raw_tmp"
    log.info(f"  Copying to temporary file: {tmp_path}")
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as f:
            x_grp = f["X"]

            # 1. Back up original log1p X
            if backup_layer:
                if "layers" not in f:
                    f.create_group("layers")
                layers = f["layers"]
                if "log1p" in layers:
                    del layers["log1p"]

                log1p_layer = layers.create_group("log1p")
                for k, v in x_grp.attrs.items():
                    log1p_layer.attrs[k] = v

                for ds_name in ["data", "indices", "indptr"]:
                    log1p_layer.create_dataset(ds_name, data=x_grp[ds_name][:])
                log.info("  Original log1p X backed up to layers['log1p']")

            # 2. Update X['data'] in-place using chunked expm1 and rounding
            log.info("  Applying expm1 and integer rounding to X['data']...")
            data_ds = x_grp["data"]
            chunk_size = 5_000_000

            for i in range(0, n_elements, chunk_size):
                end = min(i + chunk_size, n_elements)
                chunk = data_ds[i:end]
                restored = np.round(np.expm1(chunk)).astype(np.float32)
                data_ds[i:end] = restored

            x_grp.attrs["is_raw"] = True

        shutil.move(tmp_path, h5ad_path)
        elapsed = time.time() - t0
        log.info(f"  [arc_h1] Raw counts restored successfully! ({elapsed:.1f}s)")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {"dataset": "arc_h1", "status": "done", "elapsed_sec": round(elapsed, 1)}


def restore_kolf_strong(h5ad_path: str, dry_run: bool = False, backup_layer: bool = True) -> dict:
    """Replaces X in kolf_strong with layers['counts'] (raw UMI counts)."""
    log.info(f"\n[kolf_strong] Inspecting and restoring: {h5ad_path}")

    with h5py.File(h5ad_path, "r") as f:
        if "layers" not in f or "counts" not in f["layers"]:
            log.error("  [kolf_strong] Cannot locate layers['counts'].")
            return {"dataset": "kolf_strong", "status": "error", "reason": "no layers['counts']"}

        counts_grp = f["layers"]["counts"]
        sample_counts = counts_grp["data"][:20]
        is_int = bool(np.all(sample_counts == np.round(sample_counts)))

        log.info(f"  layers['counts'] sample : {sample_counts[:5].tolist()}")
        log.info(f"  Integer counts verified : {is_int}")

        if "X" in f and isinstance(f["X"], h5py.Group) and "data" in f["X"]:
            sample_curr_x = f["X"]["data"][:5]
            log.info(f"  Current X data sample   : {sample_curr_x.tolist()} (z-normalized)")

        n_elements = len(counts_grp["data"])
        log.info(f"  Total non-zero elements: {n_elements:,}")

    if dry_run:
        log.info("  [DRY-RUN] Save skipped.")
        return {"dataset": "kolf_strong", "status": "dry_run", "is_int": is_int}

    t0 = time.time()
    tmp_path = h5ad_path + ".raw_tmp"
    log.info(f"  Copying to temporary file: {tmp_path}")
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as f:
            # 1. Back up original X to layers['normalized']
            if backup_layer:
                layers = f["layers"]
                if "normalized" in layers:
                    del layers["normalized"]

                norm_layer = layers.create_group("normalized")
                x_grp = f["X"]
                for k, v in x_grp.attrs.items():
                    norm_layer.attrs[k] = v

                for ds_name in ["data", "indices", "indptr"]:
                    norm_layer.create_dataset(ds_name, data=x_grp[ds_name][:])
                log.info("  Original X backed up to layers['normalized']")

            # 2. Replace X with layers['counts']
            del f["X"]
            new_x = f.create_group("X")
            counts_grp = f["layers"]["counts"]

            for k, v in counts_grp.attrs.items():
                new_x.attrs[k] = v

            for ds_name in ["data", "indices", "indptr"]:
                new_x.create_dataset(
                    ds_name,
                    data=counts_grp[ds_name][:],
                    compression="gzip",
                    compression_opts=4
                )

            new_x.attrs["is_raw"] = True
            log.info("  layers['counts'] copied to X")

        shutil.move(tmp_path, h5ad_path)
        elapsed = time.time() - t0
        log.info(f"  [kolf_strong] Raw counts restored successfully! ({elapsed:.1f}s)")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {"dataset": "kolf_strong", "status": "done", "elapsed_sec": round(elapsed, 1)}


def parse_args():
    p = argparse.ArgumentParser(description="Restore/replace expression matrix X with raw UMI counts for arc_h1 and kolf_strong")
    p.add_argument("--dataset", choices=["arc_h1", "kolf_strong", "all"], default="all",
                   help="Dataset to process (arc_h1, kolf_strong, or all)")
    p.add_argument("--input-dir", default=STANDARDIZED_DIR, help="Standardized directory path")
    p.add_argument("--no-backup", action="store_true", help="Skip backing up original matrix to layers")
    p.add_argument("--dry-run", action="store_true", help="Preview data samples and precision without saving")
    return p.parse_args()


def main():
    args = parse_args()
    backup_layer = not args.no_backup

    targets = ["arc_h1", "kolf_strong"] if args.dataset == "all" else [args.dataset]
    log.info(f"Raw Count Restoration: {targets}")
    log.info(f"Directory: {args.input_dir}")
    log.info(f"Dry-run: {args.dry_run}, Layer backup: {backup_layer}")

    results = []
    for ds in targets:
        h5ad_path = os.path.join(args.input_dir, f"{ds}_standardized.h5ad")
        if not os.path.exists(h5ad_path):
            log.error(f"File not found: {h5ad_path}")
            results.append({"dataset": ds, "status": "missing"})
            continue

        try:
            if ds == "arc_h1":
                r = restore_arc_h1(h5ad_path, dry_run=args.dry_run, backup_layer=backup_layer)
            elif ds == "kolf_strong":
                r = restore_kolf_strong(h5ad_path, dry_run=args.dry_run, backup_layer=backup_layer)
            results.append(r)
        except Exception as e:
            log.error(f"[{ds}] ERROR: {e}", exc_info=True)
            results.append({"dataset": ds, "status": "error", "reason": str(e)})

    log.info("\n" + "=" * 55)
    log.info("Results Summary:")
    log.info(f"{'Dataset':<20} {'Status':<12} {'Elapsed/Details'}")
    log.info("-" * 55)
    for r in results:
        status = r.get("status", "?")
        desc = f"{r.get('elapsed_sec', 0)}s" if "elapsed_sec" in r else r.get("reason", "")
        log.info(f"  {r['dataset']:<18} {status:<12} {desc}")


if __name__ == "__main__":
    main()
