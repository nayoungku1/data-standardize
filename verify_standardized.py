"""
verify_standardized.py
----------------------
표준화된 h5ad 파일들이 제대로 처리되었는지 검증:
  1. 각 파일이 존재하는지
  2. 각 파일의 gene 목록이 gene_names.csv 서브셋인지
  3. gene 순서가 맞는지 (상대 순서 일치)
  4. 놓친 gene (겹쳐야 했는데 없는 gene) 확인
  5. datasets_meta.json에 있는 모든 데이터셋 처리됐는지 확인
"""

import json
import os
import sys
import pandas as pd
import anndata as ad
import numpy as np

META_JSON   = "/mnt/nas2/projects/vcc-data/datasets_meta.json"
GENE_CSV    = "/mnt/nas2/projects/vcc-data/vcc-2026/gene_names.csv"
OUTPUT_DIR  = "/mnt/nas2/projects/vcc-data/standardized"

print("=" * 70)
print("VCC 표준화 파일 검증 스크립트")
print("=" * 70)

# 1. 표준 gene 목록 로드
std_genes_df = pd.read_csv(GENE_CSV)
std_genes    = std_genes_df.iloc[:, 0].dropna().astype(str).tolist()
std_genes_set = set(std_genes)
std_rank     = {g: i for i, g in enumerate(std_genes)}
print(f"\n[표준 gene] 총 {len(std_genes):,}개 (gene_names.csv)")

# 2. datasets_meta.json 로드
with open(META_JSON) as f:
    meta = json.load(f)

print(f"[meta JSON] {len(meta)}개 데이터셋: {list(meta.keys())}")

# 3. 각 데이터셋 검증
summary_rows = []
all_ok = True

for name, entry in meta.items():
    out_path = os.path.join(OUTPUT_DIR, f"{name}_standardized.h5ad")
    
    row = {
        "dataset": name,
        "file_exists": os.path.exists(out_path),
        "n_cells": None,
        "n_overlap_genes": None,
        "expected_overlap": None,
        "gene_order_ok": None,
        "is_subset_of_std": None,
        "missing_genes": None,
        "extra_genes": None,
        "issues": []
    }
    
    if not row["file_exists"]:
        row["issues"].append("파일 없음!")
        all_ok = False
        summary_rows.append(row)
        continue
    
    # 파일 로드 (backed 모드로 메모리 절약)
    try:
        adata = ad.read_h5ad(out_path, backed="r")
    except Exception as e:
        row["issues"].append(f"파일 로드 오류: {e}")
        all_ok = False
        summary_rows.append(row)
        continue
    
    file_genes = adata.var_names.tolist()
    n_cells    = adata.n_obs
    n_genes    = adata.n_vars
    adata.file.close()
    
    row["n_cells"] = n_cells
    row["n_overlap_genes"] = n_genes
    
    # 4. gene_conversion 이용해 원본에서 예상 overlap 계산
    conv = entry.get("gene_conversion", {g: g for g in entry["gene"]})
    src_genes = entry["gene"]
    converted = [conv.get(g, g) for g in src_genes]
    
    # 중복 제거 (첫 등장 우선)
    seen = set()
    converted_unique = []
    for c in converted:
        if c not in seen:
            converted_unique.append(c)
            seen.add(c)
    
    # 표준 gene 목록과의 교집합 (표준 순서 유지)
    expected_overlap = [g for g in std_genes if g in set(converted_unique)]
    row["expected_overlap"] = len(expected_overlap)
    
    file_genes_set    = set(file_genes)
    expected_set      = set(expected_overlap)
    
    # 5. 일치 여부 확인
    missing = expected_set - file_genes_set   # 있어야 하는데 없는 gene
    extra   = file_genes_set - expected_set   # 있으면 안되는데 있는 gene
    not_in_std = file_genes_set - std_genes_set  # 표준 gene 목록에 없는 gene
    
    row["is_subset_of_std"] = len(not_in_std) == 0
    row["missing_genes"]    = len(missing)
    row["extra_genes"]      = len(extra)
    
    if missing:
        row["issues"].append(f"놓친 gene {len(missing)}개: {sorted(missing)[:10]}{'...' if len(missing)>10 else ''}")
        all_ok = False
    if extra:
        row["issues"].append(f"불필요한 gene {len(extra)}개: {sorted(extra)[:10]}{'...' if len(extra)>10 else ''}")
        all_ok = False
    if not_in_std:
        row["issues"].append(f"표준 gene 밖의 gene {len(not_in_std)}개 포함!")
        all_ok = False
    
    # 6. gene 순서 확인 (표준 순서 기준 단조증가 여부)
    file_ranks = [std_rank.get(g, -1) for g in file_genes if g in std_rank]
    if len(file_ranks) > 1:
        is_sorted = all(file_ranks[i] < file_ranks[i+1] for i in range(len(file_ranks)-1))
        row["gene_order_ok"] = is_sorted
        if not is_sorted:
            row["issues"].append("gene 순서 오류 (표준 순서 불일치)!")
            all_ok = False
    else:
        row["gene_order_ok"] = True
    
    # 7. 개수 일치
    if n_genes != len(expected_overlap):
        row["issues"].append(f"gene 수 불일치: 파일={n_genes}, 기대={len(expected_overlap)}")
        all_ok = False
    
    summary_rows.append(row)

# ============================================================
# 결과 출력
# ============================================================
print("\n" + "=" * 70)
print("검증 결과 요약")
print("=" * 70)

print(f"\n{'데이터셋':<35} {'파일':<5} {'cells':>8} {'genes':>8} {'기대':>8} {'순서':>5} {'이슈'}")
print("-" * 80)

for r in summary_rows:
    exists_mark = "OK" if r["file_exists"] else "NO"
    order_mark  = ("OK" if r["gene_order_ok"] else "NG") if r["gene_order_ok"] is not None else "-"
    issue_str   = " | ".join(r["issues"]) if r["issues"] else "OK"
    
    print(f"{r['dataset']:<35} {exists_mark:<5} "
          f"{str(r['n_cells'] or '-'):>8} "
          f"{str(r['n_overlap_genes'] or '-'):>8} "
          f"{str(r['expected_overlap'] or '-'):>8} "
          f"{order_mark:>5}  {issue_str}")

print("\n" + "=" * 70)
if all_ok:
    print("ALL PASS: 모든 데이터셋 검증 통과!")
else:
    print("FAIL: 일부 데이터셋에 문제 있음! 위 이슈를 확인하세요.")
print("=" * 70)

# ============================================================
# 상세 놓친 gene 분석
# ============================================================
print("\n\n[상세] 각 데이터셋 놓친 gene 분석")
print("-" * 70)

for name, entry in meta.items():
    out_path = os.path.join(OUTPUT_DIR, f"{name}_standardized.h5ad")
    if not os.path.exists(out_path):
        continue
    
    try:
        adata = ad.read_h5ad(out_path, backed="r")
        file_genes = set(adata.var_names.tolist())
        adata.file.close()
    except:
        continue
    
    conv = entry.get("gene_conversion", {g: g for g in entry["gene"]})
    src_genes = entry["gene"]
    converted = [conv.get(g, g) for g in src_genes]
    
    seen = set()
    converted_unique = []
    for c in converted:
        if c not in seen:
            converted_unique.append(c)
            seen.add(c)
    
    expected_overlap = set(g for g in std_genes if g in set(converted_unique))
    missing = expected_overlap - file_genes
    
    src_genes_in_std = set(converted_unique) & std_genes_set
    src_genes_NOT_in_std = set(converted_unique) - std_genes_set
    
    print(f"\n[{name}]")
    print(f"  원본 gene 수: {len(src_genes):,}  ->  변환 후 unique: {len(converted_unique):,}")
    print(f"  표준 gene과 겹치는 gene: {len(src_genes_in_std):,}  (표준에 없는 gene: {len(src_genes_NOT_in_std):,})")
    print(f"  저장된 파일의 gene 수: {len(file_genes):,}")
    if missing:
        print(f"  WARNING: 놓친 gene {len(missing)}개: {sorted(missing)[:20]}")
    else:
        print(f"  OK: 놓친 gene 없음")

print("\n검증 완료.")
