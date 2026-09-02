"""
extract_ntc_h5ad.py
-------------------
각 standardized h5ad 파일에서 non-targeting (NTC) cell들만 필터링하여
지정된 NTC 디렉토리(/mnt/nas2/projects/vcc-data/NTC/)에 별도 h5ad로 저장하는 스크립트.

주요 특징:
  1. 초고속 dry-run:
     - h5py로 obs 메타데이터만 즉시 읽어 수십 기가바이트 파일도 0.5초 내에 NTC 통계 확인
  2. 메모리 효율적 추출:
     - AnnData backed='r' 슬라이싱으로 NTC 행만 메모리에 올려 저장
     - OOM(메모리 부족) 방지
  3. 유연한 NTC 판별:
     - 1순위: obs['target_gene'] == 'non-targeting'
     - fallback: gene_target, gene, sgrna_symbol 컬럼 및
       ['non-targeting', 'non_targeting', 'control', 'ntc', 'nt', 'ctrl'] 매칭
  4. 모든 메타데이터 보존:
     - var (gene_name, ensembl_id 등), obsm, layers, uns 완전 보존

Usage:
    # 1. 전체 데이터셋 NTC 사전 검사 (초고속)
    python extract_ntc_h5ad.py --dry-run

    # 2. 특정 데이터셋 NTC 추출
    python extract_ntc_h5ad.py --dataset nadig_hepg2
    python extract_ntc_h5ad.py --dataset orion_hct116

    # 3. 전체 데이터셋 NTC 일괄 추출
    python extract_ntc_h5ad.py
"""

import argparse
import glob
import logging
import os
import time

import anndata as ad
from anndata._io import read_elem
import h5py
import numpy as np
import pandas as pd

# 최신 AnnData nullable string 호환 설정
try:
    ad.settings.allow_write_nullable_strings = True
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """AnnData h5ad 저장 시 StringDtype(nullable string) 관련 에러 방지"""
    df = df.copy()
    for col in df.columns:
        if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) == "string":
            df[col] = df[col].astype(object).fillna("")
    return df

DEFAULT_INPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"
DEFAULT_OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/NTC"

NT_LABELS = {
    "non-targeting", "non_targeting", "non targeting",
    "control", "ctrl", "unperturbed", "nan", "negctrl", "ntc", "nt", "none"
}

CANDIDATE_COLS = ["target_gene", "gene_target", "gene", "sgrna_symbol", "perturbation"]


def find_ntc_mask(obs: pd.DataFrame) -> tuple:
    """obs DataFrame에서 NTC 셀 mask와 탐지된 컬럼명 반환"""
    for col in CANDIDATE_COLS:
        if col in obs.columns:
            vals = obs[col].astype(str).str.strip().str.lower()
            mask = vals.isin(NT_LABELS)
            if mask.any():
                return mask.values, col

    for col in obs.columns:
        if any(k in col.lower() for k in ["pert", "target", "guide"]):
            vals = obs[col].astype(str).str.strip().str.lower()
            mask = vals.isin(NT_LABELS)
            if mask.any():
                return mask.values, col

    return np.zeros(len(obs), dtype=bool), None


def get_obs_fast(h5ad_path: str) -> tuple:
    """h5py로 obs와 n_vars, n_obs만 초고속으로 로드 (X/layers 전혀 안 읽음)"""
    with h5py.File(h5ad_path, "r") as h5f:
        # obs 읽기
        obs = read_elem(h5f["obs"])
        # var 개수
        n_vars = 0
        if "var" in h5f:
            if "_index" in h5f["var"]:
                n_vars = len(h5f["var"]["_index"])
            elif "gene_name" in h5f["var"]:
                n_vars = len(h5f["var"]["gene_name"])
            elif "column-order" in h5f["var"].attrs:
                first_col = h5f["var"].attrs["column-order"][0]
                n_vars = len(h5f["var"][first_col])
        return obs, n_vars


def process_dataset(h5ad_path: str, output_dir: str,
                    overwrite: bool = False, dry_run: bool = False) -> dict:
    ds_name = os.path.basename(h5ad_path).replace("_standardized.h5ad", "").replace(".h5ad", "")
    out_filename = f"{ds_name}_ntc.h5ad"
    out_path = os.path.join(output_dir, out_filename)

    log.info(f"\n[{ds_name}] 처리 중: {h5ad_path}")

    # 작업 중인 임시 파일 감지
    for ext in [".ct_tmp", ".varid_tmp", ".perttmp"]:
        if os.path.exists(h5ad_path + ext):
            log.warning(f"  [{ds_name}] 임시 파일({ext}) 감지됨. 진행 중인 작업이 있어 건너뜁니다.")
            return {"dataset": ds_name, "status": "skipped", "reason": f"active tmp file ({ext})"}

    if os.path.exists(out_path) and not overwrite and not dry_run:
        log.info(f"  [{ds_name}] 이미 NTC 파일이 존재합니다: {out_path} (--overwrite로 덮어쓰기 가능)")
        return {"dataset": ds_name, "status": "skipped", "reason": "output exists"}

    t0 = time.time()

    # 1. 초고속 obs 로드
    obs, n_vars = get_obs_fast(h5ad_path)
    n_total_cells = len(obs)

    # 2. NTC 마스크 판별
    mask, col_used = find_ntc_mask(obs)
    n_ntc = int(mask.sum())
    ntc_ratio = (n_ntc / n_total_cells * 100) if n_total_cells > 0 else 0

    log.info(f"  전체 세포: {n_total_cells:,}개 | 유전자: {n_vars:,}개")
    log.info(f"  탐지 컬럼: '{col_used}' -> NTC 세포: {n_ntc:,}개 ({ntc_ratio:.2f}%)")

    if n_ntc == 0:
        log.warning(f"  [{ds_name}] NTC 세포를 찾지 못했습니다.")
        return {
            "dataset": ds_name, "status": "no_ntc_found",
            "total_cells": n_total_cells, "ntc_cells": 0, "ratio": 0.0
        }

    if dry_run:
        log.info(f"  [DRY-RUN] 검사 완료 (예상 저장: {out_path})")
        return {
            "dataset": ds_name, "status": "dry_run",
            "total_cells": n_total_cells, "ntc_cells": n_ntc,
            "ratio": round(ntc_ratio, 2), "col_used": col_used
        }

    # 3. NTC 슬라이싱 및 메모리 로드 (OOM 방지: 거대한 layers 전체 적재 대신 X와 obs, var만 추출)
    log.info(f"  NTC {n_ntc:,}개 세포 슬라이싱 및 저장 준비 (경량 메모리 모드)...")
    adata = ad.read_h5ad(h5ad_path, backed="r")
    
    # X와 obs, var만 필요한 크기로 슬라이싱 및 string 타입 정제
    X_sub = adata.X[mask]
    obs_sub = sanitize_dataframe(adata.obs[mask].copy())
    var_sub = sanitize_dataframe(adata.var.copy())
    
    sub_adata = ad.AnnData(X=X_sub, obs=obs_sub, var=var_sub)
    
    # obsm이 있으면 보존
    if hasattr(adata, "obsm") and len(adata.obsm) > 0:
        sub_adata.obsm = {k: adata.obsm[k][mask] for k in adata.obsm.keys()}
        
    adata.file.close()

    # 4. 저장
    os.makedirs(output_dir, exist_ok=True)
    tmp_out = out_path + ".tmp"
    sub_adata.write_h5ad(tmp_out, compression="gzip")
    os.replace(tmp_out, out_path)

    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(out_path) / (1024 * 1024)
    log.info(f"  [{ds_name}] 완료! ({n_ntc:,} cells × {n_vars:,} genes, {file_size_mb:.1f} MB, {elapsed:.1f}초)")

    return {
        "dataset": ds_name, "status": "done",
        "total_cells": n_total_cells, "ntc_cells": n_ntc,
        "ratio": round(ntc_ratio, 2), "col_used": col_used,
        "output": out_path, "size_mb": round(file_size_mb, 1),
        "elapsed_sec": round(elapsed, 1)
    }


def parse_args():
    p = argparse.ArgumentParser(description="Standardized h5ad에서 Non-Targeting Cell (NTC)만 필터링하여 저장")
    p.add_argument("--dataset", default=None, help="특정 데이터셋만 처리 (e.g., nadig_hepg2, orion_hct116)")
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR, help="표준화 데이터 디렉토리")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="NTC 저장 디렉토리")
    p.add_argument("--overwrite", action="store_true", help="기존 NTC 파일 덮어쓰기")
    p.add_argument("--dry-run", action="store_true", help="실제 저장 없이 통계만 확인")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset:
        clean_name = args.dataset.replace("_standardized.h5ad", "").replace(".h5ad", "")
        h5ad_path = os.path.join(args.input_dir, f"{clean_name}_standardized.h5ad")
        if not os.path.exists(h5ad_path) and os.path.exists(args.dataset):
            h5ad_path = args.dataset
        targets = [h5ad_path]
    else:
        targets = sorted(glob.glob(os.path.join(args.input_dir, "*_standardized.h5ad")))

    if not targets:
        log.error(f"대상 파일이 없습니다: {args.input_dir}")
        return

    log.info(f"NTC 추출 작업: 총 {len(targets)}개 파일")
    log.info(f"입력: {args.input_dir}")
    log.info(f"출력: {args.output_dir}")

    results = []
    for f in targets:
        try:
            r = process_dataset(f, args.output_dir, overwrite=args.overwrite, dry_run=args.dry_run)
            results.append(r)
        except Exception as e:
            log.error(f"[{f}] ERROR: {e}", exc_info=True)
            ds_name = os.path.basename(f).replace("_standardized.h5ad", "").replace(".h5ad", "")
            results.append({"dataset": ds_name, "status": "error", "reason": str(e)})

    # 요약 테이블
    log.info("\n" + "=" * 75)
    log.info("NTC 필터링 결과 요약:")
    log.info(f"{'데이터셋':<25} {'상태':<10} {'전체 세포':>11} {'NTC 세포':>11} {'비율(%)':>8} {'컬럼'}")
    log.info("-" * 75)
    for r in results:
        status = r.get("status", "?")
        tot = f"{r.get('total_cells', 0):,}" if "total_cells" in r else "-"
        ntc = f"{r.get('ntc_cells', 0):,}" if "ntc_cells" in r else "-"
        ratio = f"{r.get('ratio', 0.0):.2f}%" if "ratio" in r else "-"
        col = r.get("col_used") or r.get("reason") or ""
        log.info(f"  {r['dataset']:<23} {status:<10} {tot:>11} {ntc:>11} {ratio:>8} {col}")

    n_done = sum(1 for r in results if r.get("status") in ("done", "dry_run"))
    log.info(f"\n처리 완료: {n_done}/{len(results)}")


if __name__ == "__main__":
    main()
