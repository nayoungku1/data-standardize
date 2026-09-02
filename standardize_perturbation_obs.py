"""
standardize_perturbation_obs.py
-------------------------------
각 standardized h5ad 파일의 obs에서:
  1. perturbation 컬럼을 'target_gene'으로 rename
     (nadig/replogle: 'gene', orion/kolf: 'gene_target', mixscale: 'gene',
      arc_h1: 'target_gene' (이미 존재, 내용만 표준화), kaggle: 'sgrna_symbol')
  2. target_gene 값을 GENCODE v32 gene symbol로 표준화 (gene_conversion 활용)
  3. non-targeting 라벨을 'non-targeting'으로 통일
     - orion_*   : CTRL -> non-targeting
     - kolf_*    : NTC  -> non-targeting
     - mixscale_*: NT   -> non-targeting
     - 공통: control, non_targeting, unperturbed, negctrl, ntc, nt, none, ctrl -> non-targeting

처리 방식:
  - shutil.copy2 + h5py 직접 수정 (X/layers/var는 건드리지 않음)
  - --dry-run 옵션으로 실제 저장 없이 변환 결과만 확인 가능
  - --overwrite 옵션으로 기존 target_gene 컬럼 강제 덮어쓰기

Usage:
    python standardize_perturbation_obs.py
    python standardize_perturbation_obs.py --dataset orion_hct116 --dry-run
    python standardize_perturbation_obs.py --overwrite
"""

import argparse
import gzip
import json
import logging
import os
import shutil

import h5py
import numpy as np
import pandas as pd
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================
# 설정
# ============================================================

META_JSON  = "/mnt/nas2/projects/vcc-data/datasets_meta.json"
GTF_GZ     = "/home/dev02/integration/gencode/gencode.v32.primary_assembly.annotation.gtf.gz"
OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"

# 표준 출력 컬럼명
STD_COL = "target_gene"

# 데이터셋별 원본 perturbation 컬럼명 (→ STD_COL로 rename)
DATASET_SRC_COL = {
    "nadig_hepg2":             "gene",
    "nadig_jurkat":            "gene",
    "replogle_rpe1":           "gene",
    "replogle_k562_gwp":       "gene",
    "replogle_k562_essential":  "gene",
    "orion_hct116":            "gene_target",
    "orion_hek293t":           "gene_target",
    "kolf_strong":             "gene_target",
    "mixscale_ifnb":           "gene",
    "mixscale_ifng":           "gene",
    "mixscale_ins":            "gene",
    "mixscale_tgfb":           "gene",
    "mixscale_tnfa":           "gene",
    "arc_h1":                  "target_gene",   # 이미 target_gene → 내용만 표준화
    "kaggle":                  "sgrna_symbol",
}

# 데이터셋별 추가 non-targeting 라벨 (소문자)
DATASET_NT_LABELS = {
    "orion_hct116":  {"ctrl"},
    "orion_hek293t": {"ctrl"},
    "kolf_strong":   {"ntc"},
    "mixscale_ifnb": {"nt"},
    "mixscale_ifng": {"nt"},
    "mixscale_ins":  {"nt"},
    "mixscale_tgfb": {"nt"},
    "mixscale_tnfa": {"nt"},
}

# 공통 non-targeting 라벨 (소문자)
COMMON_NT_LABELS = {
    "non-targeting", "non_targeting", "control", "unperturbed",
    "nan", "negctrl", "ntc", "nt", "none", "ctrl", "negative_control",
    "non targeting",
}

STANDARD_NT = "non-targeting"


# ============================================================
# 1. GENCODE v32 로드
# ============================================================

def load_gencode_v32(gtf_gz: str) -> tuple:
    """Ensembl ID → gene symbol dict, v32 gene symbol set 반환"""
    log.info(f"GENCODE v32 GTF 로딩: {gtf_gz}")
    ens2sym = {}
    v32_set = set()
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
            if gene_name:
                v32_set.add(gene_name)
            if gene_id and gene_name:
                ens2sym[gene_id] = gene_name
                ens2sym[gene_id.split(".")[0]] = gene_name
    log.info(f"  GENCODE v32: {len(v32_set):,} symbols, {len(ens2sym):,} Ensembl IDs")
    return ens2sym, v32_set


# ============================================================
# 2. 값 변환 함수
# ============================================================

def apply_conversion(val: str, ds_name: str, conv: dict,
                     ens2sym: dict, v32_set: set) -> str:
    """단일 perturbation 값을 표준화"""
    v = str(val).strip()
    v_lower = v.lower()

    # non-targeting 판별
    nt_labels = COMMON_NT_LABELS | DATASET_NT_LABELS.get(ds_name, set())
    if not v or v_lower in nt_labels:
        return STANDARD_NT

    # guide suffix 제거 (예: TP53_1 → TP53)
    base = v.rsplit("_", 1)[0] if "_" in v and v.rsplit("_", 1)[1].isdigit() else v
    if base.lower() in nt_labels:
        return STANDARD_NT

    # gene_conversion 적용
    if base in conv:
        return conv[base]
    if v in conv:
        return conv[v]

    # Ensembl ID → v32 symbol
    if base in ens2sym:
        return ens2sym[base]
    if v in ens2sym:
        return ens2sym[v]

    # 이미 v32 symbol
    if base in v32_set:
        return base

    # 변환 불가 → base 유지
    return base


def make_converter_series(raw_series: pd.Series, ds_name: str, conv: dict,
                          ens2sym: dict, v32_set: set) -> pd.Series:
    """Series 전체에 변환 적용 (unique 값만 처리해 성능 최적화)"""
    unique_vals = raw_series.unique()
    mapping = {v: apply_conversion(v, ds_name, conv, ens2sym, v32_set)
               for v in unique_vals}
    return raw_series.map(mapping)


# ============================================================
# 3. h5py로 obs 컬럼 rename + 값 표준화
# ============================================================

def write_categorical_col(obs_grp: h5py.Group, col_name: str, std_series: pd.Series):
    """h5py obs Group에 categorical 컬럼 쓰기 (anndata 0.2.0 encoding)"""
    # 기존 컬럼 삭제
    if col_name in obs_grp:
        del obs_grp[col_name]

    unique_cats = sorted(std_series.unique().tolist())
    cat_to_idx  = {c: i for i, c in enumerate(unique_cats)}

    # codes: int8이면 최대 127 categories — 초과 시 int16 사용
    n_cats = len(unique_cats)
    codes  = std_series.map(cat_to_idx).values
    codes  = codes.astype(np.int8 if n_cats <= 127 else np.int16)

    cat_grp = obs_grp.create_group(col_name)
    cat_grp.attrs["encoding-type"]    = "categorical"
    cat_grp.attrs["encoding-version"] = "0.2.0"
    cat_grp.attrs["ordered"]          = False

    dt = h5py.special_dtype(vlen=str)
    cat_grp.create_dataset("categories", data=np.array(unique_cats, dtype=dt))
    cat_grp.create_dataset("codes", data=codes)


def update_column_order(obs_grp: h5py.Group, old_col: str, new_col: str):
    """anndata column-order 속성 갱신 (old_col → new_col)"""
    if "column-order" not in obs_grp.attrs:
        return
    col_order = list(obs_grp.attrs["column-order"])
    if old_col in col_order and new_col not in col_order:
        idx = col_order.index(old_col)
        col_order[idx] = new_col
        obs_grp.attrs["column-order"] = col_order
    elif new_col not in col_order:
        col_order.append(new_col)
        obs_grp.attrs["column-order"] = col_order


# ============================================================
# 4. 단일 데이터셋 처리
# ============================================================

def process_dataset(ds_name: str, entry: dict,
                    ens2sym: dict, v32_set: set,
                    output_dir: str,
                    dry_run: bool = False,
                    overwrite: bool = False) -> dict:
    h5ad_path = os.path.join(output_dir, f"{ds_name}_standardized.h5ad")
    src_col   = DATASET_SRC_COL.get(ds_name)

    if not os.path.exists(h5ad_path):
        log.warning(f"[{ds_name}] 파일 없음: {h5ad_path}")
        return {"dataset": ds_name, "status": "missing"}
    if src_col is None:
        log.warning(f"[{ds_name}] DATASET_SRC_COL 미정의, 건너뜀")
        return {"dataset": ds_name, "status": "no_col_def"}

    log.info(f"\n[{ds_name}] 처리 중...")
    log.info(f"  파일: {h5ad_path}")
    log.info(f"  src_col='{src_col}' → STD_COL='{STD_COL}'")

    # obs 로드
    adata = ad.read_h5ad(h5ad_path, backed="r")
    obs   = adata.obs.copy()
    n_cells = len(obs)
    adata.file.close()

    if src_col not in obs.columns:
        log.warning(f"  [{ds_name}] '{src_col}' 컬럼 없음! available: {list(obs.columns)}")
        return {"dataset": ds_name, "status": "error", "reason": f"column '{src_col}' not found"}

    # arc_h1처럼 src_col == STD_COL인 경우: 이미 target_gene이 있음
    # overwrite 없이도 내용 표준화는 항상 진행
    if STD_COL in obs.columns and src_col != STD_COL and not overwrite:
        log.info(f"  [{ds_name}] '{STD_COL}' 이미 존재. --overwrite로 강제 덮어쓰기 가능.")
        return {"dataset": ds_name, "status": "skipped", "reason": "target_gene column already exists"}

    # 변환 수행
    conv      = entry.get("gene_conversion", {})
    raw_vals  = obs[src_col].astype(str).fillna("nan")
    std_vals  = make_converter_series(raw_vals, ds_name, conv, ens2sym, v32_set)

    n_nt      = int((std_vals == STANDARD_NT).sum())
    n_changed = int((raw_vals != std_vals).sum())
    n_unique  = int(std_vals.nunique())

    log.info(f"  cells={n_cells:,}, NT={n_nt:,}, 값변환={n_changed:,}, unique={n_unique:,}")

    # 변환 예시 (non-targeting 제외)
    changed   = pd.DataFrame({"raw": raw_vals, "std": std_vals})
    changed   = changed[changed.raw != changed.std].drop_duplicates()
    if not changed.empty:
        log.info(f"  변환 샘플:\n{changed.head(10).to_string(index=False)}")

    # 표준 목록에 없는 gene 경고
    if v32_set:
        non_nt_vals   = std_vals[std_vals != STANDARD_NT]
        unmapped      = set(non_nt_vals.unique()) - v32_set
        if unmapped:
            log.warning(f"  GENCODE v32 미포함 gene {len(unmapped)}개: {sorted(unmapped)[:10]}")

    if dry_run:
        log.info("  [DRY-RUN] 저장 생략")
        return {
            "dataset": ds_name, "status": "dry_run",
            "n_cells": n_cells, "n_nt": n_nt,
            "n_changed": n_changed, "n_unique": n_unique,
        }

    # h5py로 obs 수정 (tmp → 원본 덮어쓰기)
    # shutil.copyfile 사용: NAS에서 utime 권한 오류 방지 (copy2 대신)
    tmp_path = h5ad_path + ".perttmp"
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as h5f:
            obs_grp = h5f["obs"]

            # src_col과 STD_COL이 다른 경우: src_col 유지 + STD_COL 신규 추가
            # src_col == STD_COL (arc_h1): 기존 target_gene 교체
            write_categorical_col(obs_grp, STD_COL, std_vals)
            update_column_order(obs_grp, src_col, STD_COL if src_col == STD_COL else STD_COL)

        shutil.move(tmp_path, h5ad_path)
        log.info(f"  [{ds_name}] 저장 완료 → {h5ad_path}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {
        "dataset": ds_name, "status": "done",
        "n_cells": n_cells, "n_nt": n_nt,
        "n_changed": n_changed, "n_unique": n_unique,
    }


# ============================================================
# 5. 메인
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="Perturbation obs → target_gene 표준화")
    p.add_argument("--meta-json",  default=META_JSON)
    p.add_argument("--gtf-gz",     default=GTF_GZ)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--dataset",    default=None, help="특정 데이터셋만 처리")
    p.add_argument("--dry-run",    action="store_true", help="저장 없이 변환 결과 확인")
    p.add_argument("--overwrite",  action="store_true", help="기존 target_gene 컬럼 덮어쓰기")
    p.add_argument("--skip-gtf",   action="store_true", help="GTF 로딩 생략 (gene_conversion만 사용)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.meta_json) as f:
        meta = json.load(f)

    if args.skip_gtf:
        ens2sym, v32_set = {}, set()
        log.info("GTF 로딩 생략 (--skip-gtf)")
    else:
        ens2sym, v32_set = load_gencode_v32(args.gtf_gz)

    targets = {args.dataset: meta[args.dataset]} if args.dataset else meta

    results = []
    for ds_name, entry in targets.items():
        try:
            r = process_dataset(
                ds_name, entry, ens2sym, v32_set,
                output_dir=args.output_dir,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
            results.append(r)
        except Exception as e:
            log.error(f"[{ds_name}] ERROR: {e}", exc_info=True)
            results.append({"dataset": ds_name, "status": "error", "reason": str(e)})

    # 결과 요약
    log.info("\n" + "=" * 65)
    log.info("결과 요약:")
    log.info(f"{'데이터셋':<35} {'상태':<12} {'cells':>8} {'NT':>8} {'변환':>8} {'unique':>8}")
    log.info("-" * 85)
    for r in results:
        s = r.get("status", "?")
        if s in ("done", "dry_run"):
            log.info(
                f"  {r['dataset']:<35} {s:<12} {r.get('n_cells', 0):>8,} "
                f"{r.get('n_nt', 0):>8,} {r.get('n_changed', 0):>8,} "
                f"{r.get('n_unique', 0):>8,}"
            )
        else:
            log.info(f"  {r['dataset']:<35} {s:<12}  {r.get('reason', '')}")

    n_ok = sum(1 for r in results if r.get("status") in ("done", "dry_run", "skipped"))
    log.info(f"\n처리 완료: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
