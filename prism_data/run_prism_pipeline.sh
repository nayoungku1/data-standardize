#!/usr/bin/env bash
# run_prism_pipeline.sh
# -------------------------------------------------------------
# Executes post-filtering standardization for all PRISM datasets:
#   1. Gene standardization against 18,533 VCC standard genes (process_h5ad.py)
#   2. Verification (verify_standardized.py)
#   3. Perturbation column standardizing (standardize_perturbation_obs.py)
#   4. Ensembl ID addition to var (add_ensembl_id_to_var.py)
#   5. Target gene symbol updates (update_target_gene_symbols.py)
#   6. Perturbation effect calculations in uns (compute_perturbation_effects.py)
# -------------------------------------------------------------

set -euo pipefail
export HDF5_USE_FILE_LOCKING=FALSE

SCRIPT_DIR= ".."#"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STANDARDIZED_DIR="/mnt/nas2/projects/vcc-data/standardized"

PRISM_DATASETS=(
  "prism_gse261283"
  "prism_gse208240"
  "prism_gse221321"
  "prism_gse165291"
  "prism_gse150062"
  "prism_gse210681"
)

echo "================================================================="
echo "Starting PRISM Post-Filtering Standardization Pipeline"
echo "Date: $(date)"
echo "Datasets: ${PRISM_DATASETS[*]}"
echo "================================================================="

for ds in "${PRISM_DATASETS[@]}"; do
  file_path="${STANDARDIZED_DIR}/${ds}_standardized.h5ad"
  if [[ ! -f "${file_path}" ]]; then
    echo "[ERROR] File not found: ${file_path}"
    exit 1
  fi

  echo ""
  echo "-----------------------------------------------------------------"
  echo ">>> Processing Dataset: ${ds}"
  echo "-----------------------------------------------------------------"

  # 1. Gene standardization (check if already standardized to <= 18,533 genes)
  n_vars=$(python -c "import anndata as ad; a = ad.read_h5ad('${file_path}', backed='r'); print(a.n_vars); a.file.close()")
  if [[ "${n_vars}" -gt 18533 ]]; then
    echo "[Step 1] Standardizing genes against VCC standard reference (${n_vars} original genes)..."
    python "${SCRIPT_DIR}/process_h5ad.py" --dataset "${ds}" \
      --input-dir "${STANDARDIZED_DIR}" \
      --output-dir "${STANDARDIZED_DIR}" \
      --chunk-size 5000 \
      --overwrite
  else
    echo "[Step 1] Genes already standardized (${n_vars} genes <= 18,533). Skipping process_h5ad.py."
  fi

  # 2. Verification
  echo "[Step 2] Verifying standardized file..."
  python "${SCRIPT_DIR}/verify_standardized.py" --dataset "${ds}"

  # 3. Standardize perturbation column in obs
  echo "[Step 3] Standardizing perturbation obs column to 'target_gene'..."
  python "${SCRIPT_DIR}/standardize_perturbation_obs.py" --dataset "${ds}" --skip-gtf --overwrite

  # 4. Add Ensembl ID to var
  echo "[Step 4] Adding ensembl_id to var..."
  python "${SCRIPT_DIR}/add_ensembl_id_to_var.py" --dataset "${ds}" --inplace --overwrite

  # 5. Compute perturbation effects
  echo "[Step 5] Computing perturbation effects in uns..."
  python "${SCRIPT_DIR}/compute_perturbation_effects.py" --dataset "${ds}" --prism --overwrite
done

# 6. Global target gene symbol updates
echo ""
echo "-----------------------------------------------------------------"
echo ">>> [Step 6] Running global update_target_gene_symbols.py"
echo "-----------------------------------------------------------------"
python "${SCRIPT_DIR}/update_target_gene_symbols.py"

echo ""
echo "================================================================="
echo "PRISM Standardization Pipeline Completed Successfully!"
echo "Date: $(date)"
echo "================================================================="
