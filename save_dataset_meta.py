#!/usr/bin/env python3
"""
save_dataset_meta.py

Extracts metadata from an AnnData (.h5ad) file and saves/appends it to a JSON file.

Usage:
    # 1. Basic usage with auto-detection:
    python save_dataset_meta.py \
        --name mixscale_ifnb \
        --h5ad /mnt/nas2/projects/vcc-data/mixscale/h5ad/Seurat_object_IFNB_Perturb_seq.h5ad \
        --output datasets_meta.json

    # 2. Specifying fixed cell line (when not in obs):
    python save_dataset_meta.py \
        --name nadig_hepg2 \
        --h5ad /mnt/nas2/projects/vcc-data/Nadig/GSE264667_hepg2_raw_singlecell_01.h5ad \
        --cell-line HepG2 \
        --pert-col gene \
        --ctrl-label NT

    # 3. Interactive prompt mode:
    python save_dataset_meta.py --interactive
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore")

import numpy as np

# Candidate column names for automatic detection
PERT_COL_CANDIDATES = [
    "target_gene",
    "gene",
    "perturbation",
    "pert",
    "gene_target",
    "guide_target",
    "gene_target_str",
    "perturbation_name",
    "gene_symbol",
    "target",
]

CELL_LINE_COL_CANDIDATES = [
    "cell_line",
    "cell_lines",
    "cell_type",
    "celltype",
    "celltype_str",
    "cell_types",
    "cell",
    "donor",
    "tissue",
    "sample",
]

CTRL_LABEL_CANDIDATES = [
    "NT",
    "non-targeting",
    "control",
    "ctrl",
    "non_targeting",
    "NTC",
    "unperturbed",
    "Control",
    "CTRL",
    "control_gene",
    "safe_targeting",
]

GENE_NAME_VAR_CANDIDATES = [
    "gene_symbols",
    "gene_symbol",
    "gene_name",
    "gene_names",
    "symbol",
    "gene",
    "feature_name",
]

NA_STRINGS = {"n/a", "na", "none", "nan", "null", "no", ""}


def load_adata(h5ad_path: str):
    """Load AnnData in backed mode if possible, else standard load."""
    import anndata as ad
    try:
        adata = ad.read_h5ad(h5ad_path, backed="r")
        return adata, True
    except Exception as e:
        print(f"[*] Backed read failed ({e}), loading full AnnData...")
        import scanpy as sc
        adata = sc.read_h5ad(h5ad_path)
        return adata, False


def detect_or_ask_pert(
    available_cols: List[str],
    user_override: Optional[str] = None,
) -> str:
    """Finds perturbation column from candidates or asks user interactively."""
    if user_override:
        if user_override in available_cols:
            return user_override
        else:
            print(f"[!] Warning: Specified perturbation column '{user_override}' not found in obs.")
            print(f"    Available obs columns: {available_cols}")

    # Auto detect from candidates
    for cand in PERT_COL_CANDIDATES:
        if cand in available_cols:
            print(f"[+] Auto-detected perturbation column: '{cand}'")
            return cand

    # Interactive selection
    print(f"\n[?] Could not auto-detect perturbation column.")
    print(f"    Available obs columns: {available_cols}")
    
    while True:
        try:
            choice = input("Enter column name for perturbation: ").strip()
        except EOFError:
            choice = ""

        if not choice:
            print("Perturbation column is required. Please enter a valid column name.")
            continue

        if choice in available_cols:
            return choice
        else:
            print(f"'{choice}' is not in obs columns. Available: {available_cols}")


def detect_or_ask_cell_line(
    available_cols: List[str],
    user_override_col: Optional[str] = None,
    user_override_val: Optional[str] = None,
) -> Tuple[Optional[str], Optional[List[str]]]:
    """
    Resolves cell line info.
    Returns (cell_line_col, unique_cell_line_list_or_None).
    """
    # 1. Explicit cell line value passed via CLI (e.g. --cell-line HepG2 or --cell-line N/A)
    if user_override_val is not None:
        val_clean = user_override_val.strip()
        if val_clean.lower() in NA_STRINGS:
            print("[+] Cell line set to None (N/A).")
            return None, None
        print(f"[+] Using fixed cell line: '{val_clean}'")
        return None, [val_clean]

    # 2. Explicit column passed via CLI (e.g. --cell-line-col cell_type)
    if user_override_col:
        if user_override_col in available_cols:
            return user_override_col, None
        else:
            print(f"[!] Warning: Specified cell line column '{user_override_col}' not found in obs.")
            print(f"    Available obs columns: {available_cols}")

    # 3. Auto-detect from candidates
    for cand in CELL_LINE_COL_CANDIDATES:
        if cand in available_cols:
            print(f"[+] Auto-detected cell line column: '{cand}'")
            return cand, None

    # 4. Interactive fallback: user can input a column name, a fixed string (e.g. HepG2), or N/A
    print(f"\n[?] Could not auto-detect cell line column in obs.")
    print(f"    Available obs columns: {available_cols}")
    print("    • Type an obs column name to extract unique cell lines")
    print("    • Type a fixed cell line name (e.g. 'HepG2')")
    print("    • Type 'N/A' or press Enter to set as None")

    try:
        choice = input("Enter cell line (column name / fixed name / N/A): ").strip()
    except EOFError:
        choice = ""

    if not choice or choice.lower() in NA_STRINGS:
        print("[+] Cell line set to None.")
        return None, None

    if choice in available_cols:
        print(f"[+] Using obs column '{choice}' for cell line.")
        return choice, None
    else:
        print(f"[+] Using fixed cell line: '{choice}'")
        return None, [choice]


def detect_or_ask_ctrl(
    unique_perts: List[str],
    user_override: Optional[str] = None,
) -> Optional[str]:
    """Detects control / non-targeting label from perturbation values or asks user."""
    if user_override:
        if user_override.lower() in NA_STRINGS:
            return None
        return user_override

    # Check exact match
    for cand in CTRL_LABEL_CANDIDATES:
        if cand in unique_perts:
            print(f"[+] Auto-detected non-targeting label: '{cand}'")
            return cand

    # Check case-insensitive / substring match
    for cand in CTRL_LABEL_CANDIDATES:
        for p in unique_perts:
            if str(p).strip().lower() == cand.lower():
                print(f"[+] Auto-detected non-targeting label: '{p}'")
                return str(p)

    print("\n[?] Could not auto-detect non-targeting / control label.")
    sample_perts = unique_perts[:10]
    print(f"    Sample perturbation values: {sample_perts}")
    print("    (Type 'N/A' or press Enter if no control label)")

    try:
        choice = input("Enter non-targeting / control label: ").strip()
    except EOFError:
        choice = ""

    if choice.lower() in NA_STRINGS:
        return None
    return choice if choice else None


def check_is_raw(adata, sample_size: int = 500) -> bool:
    """Checks whether expression matrix X is raw integer counts or log/normalized floats."""
    try:
        # Check uns first
        if hasattr(adata, "uns") and adata.uns is not None and "log1p" in adata.uns:
            # uns['log1p'] exists, likely normalized
            return False

        # Inspect slice of X
        n_obs = min(sample_size, adata.n_obs)
        n_vars = min(sample_size, adata.n_vars)
        chunk = adata.X[:n_obs, :n_vars]
        if hasattr(chunk, "toarray"):
            chunk = chunk.toarray()
        elif hasattr(chunk, "A"):
            chunk = chunk.A

        non_zero = chunk[chunk != 0]
        if len(non_zero) == 0:
            return True

        is_integer = bool(np.all(np.equal(np.mod(non_zero, 1), 0)))
        return is_integer
    except Exception as e:
        print(f"[!] Warning: Failed to inspect raw counts: {e}")
        return True


def extract_gene_list(adata) -> List[str]:
    """Extract gene names from var_names or var dataframe."""
    var_cols = list(adata.var.columns)
    for cand in GENE_NAME_VAR_CANDIDATES:
        if cand in var_cols:
            return [str(x) for x in adata.var[cand].tolist()]
    return [str(x) for x in adata.var_names.tolist()]


def extract_dataset_metadata(
    h5ad_path: str,
    pert_col: Optional[str] = None,
    cell_line_col: Optional[str] = None,
    cell_line_val: Optional[str] = None,
    ctrl_label: Optional[str] = None,
    is_raw_override: Optional[bool] = None,
) -> Dict[str, Any]:
    """Extract metadata dictionary from an h5ad file."""
    abs_h5ad_path = os.path.abspath(h5ad_path)
    if not os.path.exists(abs_h5ad_path):
        raise FileNotFoundError(f"File not found: {abs_h5ad_path}")

    print(f"\n[*] Reading h5ad: {abs_h5ad_path}")
    adata, is_backed = load_adata(abs_h5ad_path)

    n_cell = int(adata.n_obs)
    n_gene = int(adata.n_vars)
    obs_cols = list(adata.obs.columns)
    var_cols = list(adata.var.columns)
    layers_keys = list(adata.layers.keys()) if hasattr(adata, "layers") and adata.layers is not None else []
    obsm_keys = list(adata.obsm.keys()) if hasattr(adata, "obsm") and adata.obsm is not None else []

    # 1. Gene list
    gene_list = extract_gene_list(adata)

    # 2. Cell line column & unique cell lines
    resolved_cell_line_col, resolved_cell_lines = detect_or_ask_cell_line(
        available_cols=obs_cols,
        user_override_col=cell_line_col,
        user_override_val=cell_line_val,
    )
    if resolved_cell_line_col:
        unique_cell_lines = sorted([str(x) for x in adata.obs[resolved_cell_line_col].dropna().unique().tolist()])
    else:
        unique_cell_lines = resolved_cell_lines  # can be list of strings or None

    # 3. Perturbation column & unique perturbations
    resolved_pert_col = detect_or_ask_pert(
        available_cols=obs_cols,
        user_override=pert_col,
    )
    unique_perts = sorted([str(x) for x in adata.obs[resolved_pert_col].dropna().unique().tolist()])

    # 4. Non-targeting / control label
    resolved_ctrl_label = detect_or_ask_ctrl(
        unique_perts=unique_perts,
        user_override=ctrl_label,
    )

    # 5. Is raw counts
    if is_raw_override is not None:
        is_raw = is_raw_override
    else:
        is_raw = check_is_raw(adata)

    # Close backed file if opened
    if is_backed and hasattr(adata, "file") and hasattr(adata.file, "close"):
        try:
            adata.file.close()
        except Exception:
            pass

    metadata = {
        "h5ad_path": abs_h5ad_path,
        "pert_col": resolved_pert_col,
        "cell_line_col": resolved_cell_line_col,
        "obs_columns": obs_cols,
        "var_columns": var_cols,
        "layers": layers_keys,
        "obsm": obsm_keys,
        "gene": gene_list,
        "n_cell": n_cell,
        "n_gene": n_gene,
        "unique_cell_line": unique_cell_lines,
        "unique_perturbation": unique_perts,
        "non_targeting": resolved_ctrl_label,
        "is_raw": is_raw,
    }

    return metadata


def save_to_json(
    dataset_name: str,
    metadata: Dict[str, Any],
    json_path: str,
):
    """Save or update dataset metadata in the target JSON file."""
    json_file = Path(json_path)
    data: Dict[str, Any] = {}

    if json_file.exists():
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[*] Loaded existing JSON ({len(data)} datasets) from: {json_path}")
        except Exception as e:
            print(f"[!] Warning: Could not parse existing JSON ({e}). Creating new.")

    # Update or add dataset entry
    data[dataset_name] = metadata

    # Ensure parent dir exists
    json_file.parent.mkdir(parents=True, exist_ok=True)

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[✓] Successfully saved '{dataset_name}' metadata to: {json_file.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract metadata from an AnnData (.h5ad) file and save to a JSON registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-n", "--name",
        type=str,
        help="Dataset identifier key (e.g. 'mixscale_ifnb'). If omitted in interactive mode, prompts for it.",
    )
    parser.add_argument(
        "-i", "--h5ad",
        type=str,
        help="Path to the input .h5ad file.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="datasets_meta.json",
        help="Path to output JSON file to create/update.",
    )
    parser.add_argument(
        "-p", "--pert-col",
        type=str,
        default=None,
        help="Column name in adata.obs representing perturbations/target genes.",
    )
    parser.add_argument(
        "-c", "--cell-line-col", "--celltype-col",
        dest="cell_line_col",
        type=str,
        default=None,
        help="Column name in adata.obs representing cell line (e.g. 'cell_line', 'cell_type').",
    )
    parser.add_argument(
        "--cell-line", "--celltype",
        dest="cell_line_val",
        type=str,
        default=None,
        help="Fixed cell line name (e.g. 'HepG2') or 'N/A' if no cell line info exists.",
    )
    parser.add_argument(
        "--ctrl-label",
        type=str,
        default=None,
        help="Label representing non-targeting control (e.g. 'NT', 'control'). Type 'N/A' if none.",
    )
    parser.add_argument(
        "--is-raw",
        type=lambda x: (str(x).lower() in ["true", "1", "yes"]),
        default=None,
        help="Explicitly set is_raw flag (true/false). If omitted, automatically detects.",
    )
    parser.add_argument(
        "-I", "--interactive",
        action="store_true",
        help="Prompt interactively for dataset name and h5ad path if not provided.",
    )

    args = parser.parse_args()

    # Interactive prompt for missing required args
    name = args.name
    h5ad_path = args.h5ad

    if not name:
        if args.interactive or not sys.stdin.isatty():
            name = input("Enter dataset name key (e.g. mixscale_ifnb): ").strip()
        if not name:
            parser.error("Dataset name (--name / -n) is required.")

    if not h5ad_path:
        if args.interactive or not sys.stdin.isatty():
            h5ad_path = input("Enter path to .h5ad file: ").strip()
        if not h5ad_path:
            parser.error("Input h5ad path (--h5ad / -i) is required.")

    # Extract metadata
    metadata = extract_dataset_metadata(
        h5ad_path=h5ad_path,
        pert_col=args.pert_col,
        cell_line_col=args.cell_line_col,
        cell_line_val=args.cell_line_val,
        ctrl_label=args.ctrl_label,
        is_raw_override=args.is_raw,
    )

    # Print summary
    print("\n" + "=" * 50)
    print(f"Dataset Key: {name}")
    print(f"  • h5ad_path: {metadata['h5ad_path']}")
    print(f"  • pert_col: {metadata['pert_col']}")
    print(f"  • cell_line_col: {metadata['cell_line_col']}")
    print(f"  • obs_columns ({len(metadata['obs_columns'])}): {metadata['obs_columns']}")
    print(f"  • var_columns ({len(metadata['var_columns'])}): {metadata['var_columns']}")
    print(f"  • layers: {metadata['layers']}")
    print(f"  • obsm: {metadata['obsm']}")
    print(f"  • n_cell: {metadata['n_cell']:,}")
    print(f"  • n_gene: {metadata['n_gene']:,}")
    print(f"  • unique_cell_line: {metadata['unique_cell_line']}")
    print(f"  • unique_perturbation ({len(metadata['unique_perturbation'])}): sample -> {metadata['unique_perturbation'][:10]}")
    print(f"  • non_targeting: {metadata['non_targeting']}")
    print(f"  • is_raw: {metadata['is_raw']}")
    print("=" * 50 + "\n")

    # Save to JSON
    save_to_json(
        dataset_name=name,
        metadata=metadata,
        json_path=args.output,
    )


if __name__ == "__main__":
    main()
