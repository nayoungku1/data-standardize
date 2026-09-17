#!/usr/bin/env bash
set -euo pipefail

# Directory and file settings
DATA_DIR="/mnt/nas2/projects/vcc-data/PRISM/Datasets"
OUTPUT_JSON="/mnt/nas2/projects/vcc-data/datasets_meta.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of target on-mechanism (CRISPRi) GSEs from PRISM-for-VCC2026.md
GSES=(
  "GSE210681"
  "GSE221321"
  #"GSE120861"
  #"GSE212396"
  "GSE208240"
  #"GSE146194"
  #"GSE205310"
  "GSE150062"
  "GSE261283"
  #"GSE133344"
  #"GSE250378"
  "GSE165291"
  #"GSE272093"
  #"GSE182308"
  #"GSE278572"
  #"GSE124703"
  #"GSE132080"
  #"GSE217812"
  #"GSE261025"
)

echo "=================================================="
echo "Starting metadata extraction for ${#GSES[@]} PRISM datasets"
echo "Output JSON: ${OUTPUT_JSON}"
echo "=================================================="

# Option: Set FORCE=1 to re-process datasets already in JSON
FORCE="${FORCE:-0}"

for gse in "${GSES[@]}"; do
  h5ad_file="${DATA_DIR}/${gse}.h5ad"
  name="prism_${gse,,}"  # Convert to lowercase, e.g. prism_gse210681

  if [[ ! -f "${h5ad_file}" ]]; then
    echo "[!] Warning: File not found: ${h5ad_file}. Skipping..."
    continue
  fi

  # Skip if already exists in JSON and not in FORCE mode
  if [[ "${FORCE}" != "1" ]] && [[ -f "${OUTPUT_JSON}" ]] && grep -q "\"${name}\":" "${OUTPUT_JSON}"; then
    echo "[*] Already processed '${name}' in ${OUTPUT_JSON}. Skipping (set FORCE=1 to overwrite)..."
    continue
  fi

  # Dataset-specific arguments
  extra_args=()
  if [[ "${gse}" == "GSE165291" ]]; then
    # GSE165291 uses AAVS1 (safe harbor control)
    extra_args+=(--ctrl-label "AAVS1")
  fi

  echo ""
  echo ">>> Processing [${name}] from ${h5ad_file} ..."
  if ! python "../save_dataset_meta.py" \
    --name "${name}" \
    --h5ad "${h5ad_file}" \
    --output "${OUTPUT_JSON}" \
    "${extra_args[@]}" < /dev/null; then
    echo "[!] Error: Failed to process ${name}. Continuing to next dataset..."
  fi
done

echo ""
echo "=================================================="
echo "PRISM metadata processing finished!"
echo "Saved to: ${OUTPUT_JSON}"
echo "=================================================="
