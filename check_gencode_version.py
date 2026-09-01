"""
Check overlap ratio between GENCODE GTF gene names and VCC gene_names.csv.

Usage:
    python check_gencode_version.py --version v32
    python check_gencode_version.py --version v38
    python check_gencode_version.py --version v44
    python check_gencode_version.py --version v32 v38 v44  # compare multiple
"""

import argparse
import gzip
import os
import pandas as pd
from pathlib import Path


GENCODE_GTF_PATHS = {
    "v32": "gencode/gencode.v32.primary_assembly.annotation.gtf.gz",
    "v38": "gencode/gencode.v38.primary_assembly.annotation.gtf.gz",
    "v44": "gencode/gencode.v44.primary_assembly.annotation.gtf.gz",
}

GTF_DIR = "gencode"


def get_gtf_path(version: str) -> str:
    """Return the GTF path for a given version string (e.g. 'v50')."""
    if version in GENCODE_GTF_PATHS:
        return GENCODE_GTF_PATHS[version]
    return f"{GTF_DIR}/gencode.{version}.primary_assembly.annotation.gtf.gz"

VCC_GENE_NAMES_PATH = "/mnt/nas2/projects/vcc-data/vcc-2026/gene_names.csv"


def extract_gene_names_from_gtf(gtf_gz_path: str) -> set:
    """Extract unique gene_name values from a gzipped GTF file (gene-level rows only)."""
    gene_names = set()
    print(f"  Parsing {Path(gtf_gz_path).name} ...")
    with gzip.open(gtf_gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 9:
                continue
            feature = fields[2]
            if feature != "gene":
                continue
            attrs = fields[8]
            # Parse gene_name attribute
            for attr in attrs.split(";"):
                attr = attr.strip()
                if attr.startswith("gene_name"):
                    # e.g. gene_name "TSPAN6"
                    gene_name = attr.split('"')[1] if '"' in attr else attr.split(" ", 1)[1]
                    gene_names.add(gene_name.strip())
                    break
    return gene_names


def load_vcc_genes(path: str) -> set:
    df = pd.read_csv(path)
    col = df.columns[0]  # 'gene_name'
    return set(df[col].dropna().tolist())


def compute_overlap(vcc_genes: set, gtf_genes: set, version: str):
    overlap = vcc_genes & gtf_genes
    only_vcc = vcc_genes - gtf_genes
    only_gtf = gtf_genes - vcc_genes

    n_vcc = len(vcc_genes)
    n_gtf = len(gtf_genes)
    n_overlap = len(overlap)

    overlap_ratio_vcc = n_overlap / n_vcc if n_vcc > 0 else 0.0
    overlap_ratio_gtf = n_overlap / n_gtf if n_gtf > 0 else 0.0

    print(f"\n{'='*55}")
    print(f"  GENCODE {version}")
    print(f"{'='*55}")
    print(f"  VCC genes          : {n_vcc:,}")
    print(f"  GENCODE genes      : {n_gtf:,}")
    print(f"  Overlap            : {n_overlap:,}")
    print(f"  Overlap / VCC      : {overlap_ratio_vcc:.4f}  ({overlap_ratio_vcc*100:.2f}%)")
    print(f"  Overlap / GENCODE  : {overlap_ratio_gtf:.4f}  ({overlap_ratio_gtf*100:.2f}%)")
    print(f"  Only in VCC        : {len(only_vcc):,}")
    print(f"  Only in GENCODE    : {len(only_gtf):,}")

    if only_vcc:
        preview = sorted(only_vcc)[:10]
        print(f"  Missing from GENCODE (first 10): {preview}")

    return {
        "version": version,
        "n_vcc": n_vcc,
        "n_gtf": n_gtf,
        "n_overlap": n_overlap,
        "overlap_ratio_vcc": overlap_ratio_vcc,
        "overlap_ratio_gtf": overlap_ratio_gtf,
        "n_only_vcc": len(only_vcc),
        "n_only_gtf": len(only_gtf),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check overlap between GENCODE gene names and VCC gene_names.csv"
    )
    parser.add_argument(
        "--version",
        nargs="+",
        default=list(GENCODE_GTF_PATHS.keys()),
        metavar="VERSION",
        help="GENCODE version(s) to check (e.g. v32 v38 v44 v50). Default: all pre-defined versions.",
    )
    args = parser.parse_args()

    print(f"\nLoading VCC gene list from: {VCC_GENE_NAMES_PATH}")
    vcc_genes = load_vcc_genes(VCC_GENE_NAMES_PATH)
    print(f"  Loaded {len(vcc_genes):,} VCC genes.")

    results = []
    for version in args.version:
        gtf_path = get_gtf_path(version)
        if not os.path.exists(gtf_path):
            print(f"\n[WARNING] GTF file not found for {version}: {gtf_path}")
            continue
        gtf_genes = extract_gene_names_from_gtf(gtf_path)
        result = compute_overlap(vcc_genes, gtf_genes, version)
        results.append(result)

    if len(results) > 1:
        print(f"\n{'='*55}")
        print("  Summary Comparison")
        print(f"{'='*55}")
        df = pd.DataFrame(results).set_index("version")
        print(df[["n_vcc", "n_gtf", "n_overlap", "overlap_ratio_vcc", "overlap_ratio_gtf",
                   "n_only_vcc", "n_only_gtf"]].to_string())

    print()


if __name__ == "__main__":
    main()