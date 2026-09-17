"""
add_ensembl_id_to_var.py
------------------------
Adds an 'ensembl_id' column to var in each standardized h5ad file or arbitrary h5ad file.

Queries Ensembl IDs from the GENCODE v32 GTF based on var's index
(gene_name, GENCODE v32 symbol).

- Unmappable genes are assigned an empty string ('')
- All existing var columns are preserved
- Adds the dataset directly to the var group using h5py (X/layers untouched)
- Encodes with AnnData 0.2.0 string-array specification to prevent byte-string decoding issues
- By default, leaves the input raw file UNTOUCHED and writes the standardized result to --output-file or --output-dir.
- Use --inplace only if you explicitly want to modify the input file directly.
- --dry-run: Checks mapping coverage without saving
- --overwrite: Overwrites existing output file or ensembl_id column

Usage:
    # Process raw h5ad and save to standardized directory (Original file untouched):
    python add_ensembl_id_to_var.py --h5ad /mnt/nas2/projects/vcc-data/vcc-2026/context_A.h5ad

    # Process raw h5ad and save to specific output path:
    python add_ensembl_id_to_var.py --h5ad /path/to/custom.h5ad --output-file /path/to/custom_standardized.h5ad

    # In-place modification:
    python add_ensembl_id_to_var.py --h5ad /path/to/custom.h5ad --inplace --overwrite

    # Batch / dataset lookup in standardized directory:
    python add_ensembl_id_to_var.py
    python add_ensembl_id_to_var.py --dataset nadig_hepg2
"""

import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import gzip
import json
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

# ============================================================
# Configuration
# ============================================================

META_JSON  = "/mnt/nas2/projects/vcc-data/datasets_meta.json"
GTF_GZ     = "/home/dev02/integration/gencode/gencode.v32.primary_assembly.annotation.gtf.gz"
OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"


# ============================================================
# 1. Build GENCODE v32 → gene_name: ensembl_id Mapping
# ============================================================

def load_v32_name_to_ensembl(gtf_gz: str) -> dict:
    """
    Parses gene_name -> ensembl_id (versionless ENSG...) mapping from GENCODE v32 GTF.
    In case of multiple Ensembl IDs for the same gene_name, uses the first occurrence.
    """
    log.info(f"Loading GENCODE v32 GTF: {gtf_gz}")
    name2ens = {}
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
            if gene_name and gene_id:
                ensg = gene_id.split(".")[0]  # Remove version: ENSG00000123456.7 -> ENSG00000123456
                if gene_name not in name2ens:
                    name2ens[gene_name] = ensg

    log.info(f"  Mapping built: {len(name2ens):,} gene symbols -> Ensembl IDs")
    return name2ens


# ============================================================
# 2. Helper for Inspecting h5py var Structure
# ============================================================

def get_var_gene_names(h5f: h5py.File) -> list:
    """Returns gene names (index order) from the var group."""
    var = h5f["var"]
    idx_name = "_index" if "_index" in var else var.attrs.get("_index", None)
    if idx_name and idx_name in var:
        ds = var[idx_name]
        if isinstance(ds, h5py.Group):
            if "values" in ds:
                return [v.decode() if isinstance(v, bytes) else str(v) for v in ds["values"][:]]
            elif "codes" in ds:
                codes = ds["codes"][:]
                cats  = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][:]]
                return [cats[c] for c in codes]
        else:
            return [v.decode() if isinstance(v, bytes) else str(v) for v in ds[:]]
    raise ValueError(f"Cannot locate gene index in var. keys={list(var.keys())}, attrs={dict(var.attrs)}")


# ============================================================
# 3. Single Dataset Processing
# ============================================================

def process_dataset(ds_name: str, input_h5ad: str, output_h5ad: str,
                    name2ens: dict,
                    dry_run: bool = False,
                    overwrite: bool = False) -> dict:
    log.info(f"\n[{ds_name}] Processing: {input_h5ad}")
    if input_h5ad != output_h5ad:
        log.info(f"  Target output: {output_h5ad}")

    if not os.path.exists(input_h5ad):
        log.warning(f"  [{ds_name}] Input file does not exist: {input_h5ad}")
        return {"dataset": ds_name, "status": "missing"}

    if os.path.exists(output_h5ad) and input_h5ad != output_h5ad and not overwrite:
        log.info(f"  [{ds_name}] Output file already exists: {output_h5ad}. Use --overwrite to replace.")
        return {"dataset": ds_name, "status": "skipped", "reason": "output file exists"}

    with h5py.File(input_h5ad, "r") as h5f:
        # Check existing ensembl_id column if in-place
        if "ensembl_id" in h5f["var"] and input_h5ad == output_h5ad and not overwrite:
            log.info(f"  [{ds_name}] 'ensembl_id' already exists in var. Use --overwrite to replace.")
            return {"dataset": ds_name, "status": "skipped", "reason": "already exists"}

        gene_names = get_var_gene_names(h5f)

    n_genes = len(gene_names)

    # Map gene_name -> ensembl_id
    ensembl_ids = [name2ens.get(g, "") for g in gene_names]

    n_mapped   = sum(1 for e in ensembl_ids if e)
    n_unmapped = n_genes - n_mapped
    coverage   = n_mapped / n_genes * 100 if n_genes > 0 else 0

    log.info(f"  genes={n_genes:,}, mapped={n_mapped:,} ({coverage:.1f}%), unmapped={n_unmapped:,}")

    if n_unmapped > 0:
        unmapped_genes = [g for g, e in zip(gene_names, ensembl_ids) if not e]
        log.warning(f"  {n_unmapped} unmapped genes: {unmapped_genes[:15]}{'...' if n_unmapped > 15 else ''}")

    if dry_run:
        log.info(f"  [DRY-RUN] Save skipped")
        return {
            "dataset": ds_name, "status": "dry_run",
            "n_genes": n_genes, "n_mapped": n_mapped,
            "n_unmapped": n_unmapped, "coverage_pct": round(coverage, 2),
        }

    # Ensure output directory exists
    out_dir = os.path.dirname(output_h5ad)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Copy input to temporary output file
    tmp_path = output_h5ad + ".varid_tmp"
    shutil.copyfile(input_h5ad, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as h5f:
            var_grp = h5f["var"]

            # Remove existing column if overwriting
            if "ensembl_id" in var_grp:
                del var_grp["ensembl_id"]

            # Save as string-array (AnnData 0.2.0 string-array specification)
            dt = h5py.string_dtype(encoding="utf-8")
            ds = var_grp.create_dataset(
                "ensembl_id",
                data=np.array(ensembl_ids, dtype=object),
                dtype=dt,
            )
            ds.attrs["encoding-type"]    = "string-array"
            ds.attrs["encoding-version"] = "0.2.0"

            # Update column-order
            if "column-order" in var_grp.attrs:
                col_order = list(var_grp.attrs["column-order"])
                if "ensembl_id" not in col_order:
                    col_order.append("ensembl_id")
                var_grp.attrs["column-order"] = col_order

        shutil.move(tmp_path, output_h5ad)
        log.info(f"  [{ds_name}] Saved successfully -> {output_h5ad}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {
        "dataset": ds_name, "status": "done",
        "n_genes": n_genes, "n_mapped": n_mapped,
        "n_unmapped": n_unmapped, "coverage_pct": round(coverage, 2),
    }


# ============================================================
# 4. Main CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Add ensembl_id column to var in standardized or custom h5ad files")
    p.add_argument("--h5ad",        default=None, help="Direct path to an input h5ad file")
    p.add_argument("--output-file", default=None, help="Explicit destination file path (leaves original input untouched)")
    p.add_argument("--output-dir",  default=OUTPUT_DIR, help="Standardized output directory path (default: /mnt/nas2/projects/vcc-data/standardized)")
    p.add_argument("--inplace",     action="store_true", help="Modify the input file directly in-place")
    p.add_argument("--meta-json",   default=META_JSON, help="Path to metadata JSON")
    p.add_argument("--gtf-gz",      default=GTF_GZ, help="Path to GENCODE v32 GTF")
    p.add_argument("--dataset",     default=None, help="Process a specific dataset name or path")
    p.add_argument("--dry-run",     action="store_true", help="Preview mapping coverage without saving")
    p.add_argument("--overwrite",   action="store_true", help="Overwrite existing output file or ensembl_id column")
    return p.parse_args()


def main():
    args = parse_args()

    name2ens = load_v32_name_to_ensembl(args.gtf_gz)
    results = []

    # Case 1: Direct h5ad path via --h5ad or --dataset pointing to an existing file
    target_h5ad = args.h5ad
    if not target_h5ad and args.dataset and os.path.exists(args.dataset):
        target_h5ad = args.dataset

    if target_h5ad:
        if not os.path.exists(target_h5ad):
            log.error(f"File not found: {target_h5ad}")
            return
        ds_name = args.dataset if (args.dataset and not os.path.exists(args.dataset)) else os.path.splitext(os.path.basename(target_h5ad))[0]

        if args.output_file:
            out_path = args.output_file
        elif args.inplace:
            out_path = target_h5ad
        else:
            clean_name = ds_name.replace("_standardized", "")
            out_path = os.path.join(args.output_dir, f"{clean_name}_standardized.h5ad")

        try:
            r = process_dataset(
                ds_name, target_h5ad, out_path, name2ens,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
            results.append(r)
        except Exception as e:
            log.error(f"[{ds_name}] ERROR: {e}", exc_info=True)
            results.append({"dataset": ds_name, "status": "error", "reason": str(e)})
    else:
        # Case 2: Use meta_json and output_dir
        if not os.path.exists(args.meta_json):
            log.error(f"Metadata JSON not found: {args.meta_json}")
            return

        with open(args.meta_json) as f:
            meta = json.load(f)

        if args.dataset:
            if args.dataset not in meta:
                h5ad_path = os.path.join(args.output_dir, f"{args.dataset}_standardized.h5ad")
                if not os.path.exists(h5ad_path):
                    h5ad_path = os.path.join(args.output_dir, f"{args.dataset}.h5ad")
                if os.path.exists(h5ad_path):
                    targets = {args.dataset: None}
                else:
                    log.error(f"Dataset '{args.dataset}' not found in {args.meta_json} nor in {args.output_dir}")
                    return
            else:
                targets = {args.dataset: meta[args.dataset]}
        else:
            targets = meta

        for ds_name in targets:
            h5ad_path = os.path.join(args.output_dir, f"{ds_name}_standardized.h5ad")
            if not os.path.exists(h5ad_path):
                alt_path = os.path.join(args.output_dir, f"{ds_name}.h5ad")
                if os.path.exists(alt_path):
                    h5ad_path = alt_path

            if not os.path.exists(h5ad_path):
                log.warning(f"[{ds_name}] File not found: {h5ad_path}")
                results.append({"dataset": ds_name, "status": "missing"})
                continue

            if os.path.exists(h5ad_path + ".perttmp"):
                log.warning(f"[{ds_name}] .perttmp file exists (active job?), skipping")
                results.append({"dataset": ds_name, "status": "skipped", "reason": ".perttmp exists"})
                continue

            out_path = args.output_file if args.output_file else h5ad_path

            try:
                r = process_dataset(
                    ds_name, h5ad_path, out_path, name2ens,
                    dry_run=args.dry_run,
                    overwrite=args.overwrite,
                )
                results.append(r)
            except Exception as e:
                log.error(f"[{ds_name}] ERROR: {e}", exc_info=True)
                results.append({"dataset": ds_name, "status": "error", "reason": str(e)})

    # Summary table
    log.info("\n" + "=" * 70)
    log.info("Results Summary:")
    log.info(f"{'Dataset':<35} {'Status':<12} {'Genes':>8} {'Mapped':>8} {'Unmapped':>8} {'Coverage':>10}")
    log.info("-" * 90)
    for r in results:
        s = r.get("status", "?")
        if s in ("done", "dry_run"):
            log.info(
                f"  {r['dataset']:<35} {s:<12} {r.get('n_genes', 0):>8,} "
                f"{r.get('n_mapped', 0):>8,} {r.get('n_unmapped', 0):>8,} "
                f"{r.get('coverage_pct', 0):>9.1f}%"
            )
        else:
            log.info(f"  {r['dataset']:<35} {s:<12}  {r.get('reason', '')}")

    n_ok = sum(1 for r in results if r.get("status") in ("done", "dry_run"))
    log.info(f"\nProcessing completed: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
