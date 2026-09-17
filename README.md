# VCC Data Standardization Pipeline

This repository contains tools and scripts for downloading reference annotations, standardizing gene symbols and perturbation metadata, formatting expression matrices, and extracting control subsets across all single-cell perturbation datasets.

---

## 1. Reference Annotation Setup

### Download GENCODE Comprehensive Primary Assembly (PRI) GTF
Specify the target GENCODE version in `{VERSION}` (e.g., `32`).
```bash
mkdir -p gencode
cd gencode
wget "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{VERSION}/gencode.v{VERSION}.primary_assembly.annotation.gtf.gz"
cd ..
```

### Inspect GENCODE Version of Reference Gene Symbols
Verify which GENCODE release matches your reference gene symbol set:
```bash
uv run python check_gencode_version.py --version v{VERSION}
```

---

## 2. Core Gene Symbol Standardization

Filter and reorder `var` gene symbols across all raw h5ad datasets to match GENCODE v32 and the official reference (`vcc-2026/gene_names.csv`):
```bash
nohup uv run python -u process_h5ad.py \
  --output-dir /mnt/nas2/projects/vcc-data/standardized \
  --chunk-size 5000 > standardize_all.log 2>&1 &
```

---

## 3. Post-Processing & Standardization Pipeline Scripts

### 1) `verify_standardized.py`
- **Purpose**: Comprehensively validates that all standardized h5ad files in `/mnt/nas2/projects/vcc-data/standardized/` match GENCODE v32 gene symbols, maintain exact relative order with `gene_names.csv`, and contain zero missing or extraneous genes.
- **Usage**:
  ```bash
  # Validate all standardized h5ad files
  python verify_standardized.py

  # Validate files in a custom directory
  python verify_standardized.py --dir /mnt/nas2/projects/vcc-data/standardized
  ```

---

### 2) `standardize_perturbation_obs.py`
- **Purpose**: Unifies perturbation target columns to `target_gene`, converts gene names to official GENCODE v32 symbols, and standardizes diverse non-targeting control labels (`CTRL`, `NTC`, `NT`, `control`, etc.) to the canonical label `non-targeting`.
  - `arc_h1`: Standardizes in-place on `target_gene`
  - `orion_*`: `gene_target` (`CTRL` -> `non-targeting`) -> `target_gene`
  - `kolf_strong`: `gene_target` (`NTC` -> `non-targeting`) -> `target_gene`
  - `mixscale_*`: `gene` (`NT` -> `non-targeting`) -> `target_gene`
  - `replogle_*`, `nadig_*`: `gene` -> `target_gene`
  - `kaggle`: `sgrna_symbol` -> `target_gene`
- **Usage**:
  ```bash
  # Dry-run preview without saving changes
  python standardize_perturbation_obs.py --dry-run --skip-gtf

  # Process a specific dataset
  python standardize_perturbation_obs.py --dataset orion_hct116

  # Process all datasets (overwrite existing target_gene column)
  python standardize_perturbation_obs.py --overwrite
  ```

---

### 3) `add_ensembl_id_to_var.py`
- **Purpose**: Maps GENCODE v32 gene symbols to versionless Ensembl IDs (`ENSG...`) using the GTF annotation and adds the `ensembl_id` column to `var`. Adheres to AnnData 0.2.0 `string-array` encoding standards to prevent byte-string decoding artifacts (`b'ENSG...'`).
- **Usage**:
  ```bash
  # Check mapping coverage (dry-run)
  python add_ensembl_id_to_var.py --dry-run

  # Apply to a specific dataset
  python add_ensembl_id_to_var.py --dataset nadig_hepg2 --overwrite

  # Apply to all datasets
  python add_ensembl_id_to_var.py --overwrite
  ```

---

### 4) `standardize_cell_type_obs.py`
- **Purpose**: Standardizes cell line / cell type annotations in `obs`. Injects a unified `cell_type` categorical column for single-cell line datasets (e.g., `HCT116`, `HEK293T`), or renames legacy `celltype` columns to `cell_type`.
- **Usage**:
  ```bash
  # Assign specific cell_type to a dataset
  python standardize_cell_type_obs.py --dataset orion_hct116 --cell-type HCT116
  python standardize_cell_type_obs.py --dataset orion_hek293t --cell-type HEK293T
  python standardize_cell_type_obs.py --dataset nadig_hepg2 --cell-type hepg2

  # Rename existing 'celltype' column in kolf_strong while preserving values
  python standardize_cell_type_obs.py --dataset kolf_strong --rename-only

  # Batch apply using preset dictionary
  python standardize_cell_type_obs.py --all --dry-run
  python standardize_cell_type_obs.py --all
  ```

---

### 5) `restore_raw_counts.py`
- **Purpose**: Restores true integer raw UMI counts into the expression matrix `X` for datasets that were previously stored as normalized or log-transformed values:
  - `arc_h1`: Converts $\ln(1 + \text{count})$ back to integer raw counts via $\text{expm1}(X)$ and integer rounding (exact precision within $1.91 \times 10^{-6}$). Original log1p values are preserved in `layers['log1p']`.
  - `kolf_strong`: Replaces normalized `X` with integer raw counts from `layers['counts']`. Original normalized matrix is preserved in `layers['normalized']`.
- **Usage**:
  ```bash
  # Preview restored count values and integer precision
  python restore_raw_counts.py --dry-run

  # Restore a specific dataset
  python restore_raw_counts.py --dataset arc_h1
  python restore_raw_counts.py --dataset kolf_strong

  # Restore both datasets
  python restore_raw_counts.py
  ```

---

### 6) `extract_ntc_h5ad.py`
- **Purpose**: Slices non-targeting control cells (`target_gene == 'non-targeting'`) from all standardized datasets and saves lightweight, standalone NTC h5ad files to `/mnt/nas2/projects/vcc-data/NTC/{dataset}_ntc.h5ad`.
  - **Memory Efficient**: Slices only `X`, `obs`, `var`, and `obsm` from backed mode to prevent out-of-memory errors on massive datasets (50GB+).
  - **Pandas / AnnData Compatibility**: Automatically handles `pd.StringDtype` nullable strings to ensure backward-compatible h5ad writing.
  - **Incremental Execution**: Automatically skips datasets whose output files already exist.
- **Usage**:
  ```bash
  # Preview NTC cell counts and proportions across all datasets
  python extract_ntc_h5ad.py --dry-run

  # Extract NTC for a specific dataset
  python extract_ntc_h5ad.py --dataset nadig_hepg2

  # Batch extract all datasets (automatically skips completed files)
  python extract_ntc_h5ad.py
  
  # Force overwrite existing NTC files
  python extract_ntc_h5ad.py --overwrite
  ```

---

### 7) `update_target_gene_symbols.py`
- **Purpose**: Directly updates target gene anomalies (Ensembl IDs or pseudogene aliases) in standardized h5ad files and metadata JSON files to official GENCODE v32 gene symbols:
  - `ENSG00000170846` $\rightarrow$ `AC093323.1` (GENCODE v32 official symbol)
  - `ENSG00000230707` $\rightarrow$ `AL589987.1` (GENCODE v32 official symbol)
  - `AHSA2` $\rightarrow$ `AHSA2P` (GENCODE v32 pseudogene symbol)
- **Usage**:
  ```bash
  # Preview target gene updates (dry-run)
  python update_target_gene_symbols.py --dry-run

  # Apply updates in-place to standardized h5ad files and metadata JSON
  python update_target_gene_symbols.py
  ```

---

### 8) `compute_perturbation_effects.py`
- **Purpose**: Computes perturbation-specific effect metrics (**logFC** and **delta**) for both normalized (Option A) and raw (Option B) counts relative to non-targeting controls (`target_gene == 'non-targeting'`) and stores them directly into `adata.uns` across standardized datasets (excluding `mixscale_*`).
  - **Metrics Computed**:
    - **Normalized (Option A)**:
      - Cell library-size normalized to 10,000 counts (CP10k) and transformed via $\ln(1 + x)$ (natural log):
        $$X^{\text{norm}} = \ln\left(1 + \frac{X}{\sum X} \times 10{,}000\right)$$
      - `delta_norm`: Difference in mean log-normalized expression:
        $$\text{delta\_norm} = \overline{X_p^{\text{norm}}} - \overline{X_{\text{ctrl}}^{\text{norm}}}$$
      - `logfc_norm`: $\log_2$ fold-change converted from natural log difference (Scanpy standard):
        $$\text{logfc\_norm} = \log_2(e) \times \text{delta\_norm} = \frac{\text{delta\_norm}}{\ln(2)}$$
    - **Raw Counts (Option B)**:
      - Computed directly on raw integer expression matrix $X$:
      - `delta_raw`: $\overline{X_p} - \overline{X_{\text{ctrl}}}$
      - `logfc_raw`: $\log_2\left(\frac{\overline{X_p} + \epsilon}{\overline{X_{\text{ctrl}}} + \epsilon}\right)$ (default $\epsilon = 0.1$)
    - **Standard Aliases**:
      - `logfc` $\rightarrow$ `logfc_norm`
      - `delta` $\rightarrow$ `delta_norm`
  - **Storage**:
    - Stored as 2D float32 matrices in `adata.uns['logfc_norm']`, `adata.uns['delta_norm']`, `adata.uns['logfc_raw']`, `adata.uns['delta_raw']` (shape: $n_{\text{perts}} \times n_{\text{genes}}$), with matching perturbation names in `adata.uns['perturbations']`.
    - Easily queried as a DataFrame via helper function `get_perturbation_effect(adata, 'logfc_norm')`.
    - Written in-place via `h5py` within seconds without rewriting $X$.
  - **Target Datasets (16 datasets, strictly excluding `mixscale_*`)**:
    - Standard (10): `kaggle`, `nadig_hepg2`, `nadig_jurkat`, `replogle_rpe1`, `replogle_k562_essential`, `replogle_k562_gwp`, `orion_hct116`, `orion_hek293t`, `arc_h1`, `kolf_strong`.
    - PRISM (6, via `--prism`): `prism_gse210681`, `prism_gse221321`, `prism_gse208240`, `prism_gse150062`, `prism_gse261283`, `prism_gse165291`.
- **Usage**:
  ```bash
  # Preview calculations on a single dataset without modifying file
  python compute_perturbation_effects.py --dataset kaggle --dry-run

  # Apply in-place to a specific dataset
  python compute_perturbation_effects.py --dataset kaggle

  # Batch process all 10 non-mixscale datasets in the background
  nohup python -u compute_perturbation_effects.py --all > compute_effects.log 2>&1 &
  ```


