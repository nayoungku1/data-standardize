#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# run_filter_prism.sh
# -------------------
# Shell runner for filter_prism_crispri.py
#
# Usage:
#   # Preview statistics (Dry-run mode)
#   ./run_filter_prism.sh --dry-run
#
#   # Run filtering for all target PRISM datasets
#   ./run_filter_prism.sh
#
#   # Run filtering in background with log
#   nohup ./run_filter_prism.sh > filter_prism.log 2>&1 &
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="/mnt/nas2/projects/vcc-data/PRISM/Datasets"
OUTPUT_DIR="/mnt/nas2/projects/vcc-data/standardized"

# Ensure HDF5 locking is disabled for NFS
export HDF5_USE_FILE_LOCKING="FALSE"

echo "=================================================="
echo "PRISM CRISPRi & Single Perturbation Filtering"
echo "Script:     ${SCRIPT_DIR}/filter_prism_crispri.py"
echo "Input Dir:  ${INPUT_DIR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Options:    $*"
echo "=================================================="

python3 -u "${SCRIPT_DIR}/filter_prism_crispri.py" \
  --input-dir "${INPUT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "$@"

echo ""
echo "=================================================="
echo "Filtering script finished!"
echo "=================================================="
