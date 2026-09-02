"""
standardize_cell_type_obs.py
----------------------------
각 standardized h5ad 파일의 obs에서 cell_type 컬럼을 표준화/추가하는 스크립트.

주요 기능:
  1. cell_type이 없는 경우: CLI로 지정한 cell_type 이름(e.g., 'hct116')으로 컬럼 생성
  2. celltype으로 되어 있는 경우 (e.g., kolf_strong):
     - --cell-type이 지정되면 해당 값으로 'cell_type' 컬럼 생성
     - 지정되지 않으면 기존 'celltype'의 값을 그대로 'cell_type'으로 rename/복사
  3. 이미 cell_type이 있는 경우 (e.g., mixscale_*, arc_h1):
     - --overwrite 옵션이 없으면 건너뜀 (안전 장치)

처리 방식:
  - AnnData Categorical 형식 (categories + codes)으로 h5py에 직접 기록
  - X / layers / var 등 거대한 행렬 데이터는 전혀 건드리지 않아 매우 빠르고 안전함
  - shutil.copyfile로 임시 파일 생성 후 안전하게 atomic rename

Usage:
    # 특정 데이터셋에 cell_type 추가
    python standardize_cell_type_obs.py --dataset orion_hct116 --cell-type hct116
    python standardize_cell_type_obs.py --dataset orion_hek293t --cell-type hek293t
    python standardize_cell_type_obs.py --dataset nadig_hepg2 --cell-type hepg2
    python standardize_cell_type_obs.py --dataset kolf_strong --cell-type kolf2.1j --overwrite

    # kolf_strong 기존 celltype 컬럼의 값을 그대로 cell_type으로 rename만 할 때:
    python standardize_cell_type_obs.py --dataset kolf_strong --rename-only

    # 기본 프리셋을 이용해 일괄 적용할 때:
    python standardize_cell_type_obs.py --all
    python standardize_cell_type_obs.py --all --dry-run
"""

import argparse
import glob
import logging
import os
import shutil

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"
TARGET_COL = "cell_type"

# 기본 권장 cell_type 매핑 프리셋 (--all 실행 시 사용)
DEFAULT_CELL_TYPES = {
    "nadig_hepg2":             "hepg2",
    "nadig_jurkat":            "jurkat",
    "replogle_rpe1":           "rpe1",
    "replogle_k562_gwp":       "k562",
    "replogle_k562_essential":  "k562",
    "orion_hct116":            "hct116",
    "orion_hek293t":           "hek293t",
    "kolf_strong":             "kolf2.1j",  # 기존 celltype: 'KRAB-ZIM3-dCas9 KOLF2.1J hiPSC'
    # arc_h1: 이미 cell_type=['ARC_H1'] 있음
    # mixscale_*: 이미 cell_type=['A549', 'BXPC3', 'HAP1'] 복수 세포주 있음
    # kaggle: 필요 시 지정
}


def get_n_cells(h5f: h5py.File) -> int:
    """obs 그룹에서 n_cells 확인"""
    obs = h5f["obs"]
    if "_index" in obs:
        return len(obs["_index"])
    for k in obs.keys():
        ds = obs[k]
        if isinstance(ds, h5py.Group) and "codes" in ds:
            return len(ds["codes"])
        elif hasattr(ds, "shape") and len(ds.shape) > 0:
            return ds.shape[0]
    # X shape fallback
    if "X" in h5f:
        if "shape" in h5f["X"].attrs:
            return int(h5f["X"].attrs["shape"][0])
    raise ValueError("n_cells를 확인할 수 없습니다.")


def update_column_order(obs_grp: h5py.Group, col_name: str, old_col: str = None):
    """AnnData obs column-order 메타데이터 갱신"""
    if "column-order" not in obs_grp.attrs:
        return
    col_order = list(obs_grp.attrs["column-order"])
    if old_col and old_col in col_order and col_name not in col_order:
        idx = col_order.index(old_col)
        col_order[idx] = col_name
    elif col_name not in col_order:
        col_order.append(col_name)
    obs_grp.attrs["column-order"] = col_order


def process_h5ad(h5ad_path: str, ds_name: str, cell_type_val: str = None,
                 rename_only: bool = False, overwrite: bool = False,
                 dry_run: bool = False) -> dict:
    log.info(f"\n[{ds_name}] 검사 중: {h5ad_path}")

    if not os.path.exists(h5ad_path):
        log.warning(f"  [{ds_name}] 파일이 존재하지 않습니다: {h5ad_path}")
        return {"dataset": ds_name, "status": "missing"}

    if os.path.exists(h5ad_path + ".perttmp"):
        log.warning(f"  [{ds_name}] 다른 작업(.perttmp)이 진행 중입니다. 건너뜁니다.")
        return {"dataset": ds_name, "status": "skipped", "reason": ".perttmp exists"}

    with h5py.File(h5ad_path, "r") as h5f:
        obs = h5f["obs"]
        obs_keys = list(obs.keys())
        n_cells = get_n_cells(h5f)

        has_cell_type = TARGET_COL in obs_keys
        has_celltype = "celltype" in obs_keys

        # 현재 상태 출력
        existing_info = []
        if has_cell_type:
            ds = obs[TARGET_COL]
            if isinstance(ds, h5py.Group) and "categories" in ds:
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][:5]]
                existing_info.append(f"cell_type categories={cats}")
            else:
                existing_info.append("cell_type (array)")
        if has_celltype:
            ds = obs["celltype"]
            if isinstance(ds, h5py.Group) and "categories" in ds:
                cats = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][:5]]
                existing_info.append(f"celltype categories={cats}")
            else:
                existing_info.append("celltype (array)")

        log.info(f"  cells: {n_cells:,}, 기존 컬럼: {existing_info if existing_info else '없음'}")

        # 처리 분기 결정
        action = None
        target_value_desc = ""

        if has_cell_type and not overwrite:
            log.info(f"  [{ds_name}] 이미 '{TARGET_COL}' 컬럼이 존재합니다. 변경하려면 --overwrite를 지정하세요.")
            return {"dataset": ds_name, "status": "skipped", "reason": "already has cell_type"}

        if rename_only:
            if not has_celltype:
                log.warning(f"  [{ds_name}] --rename-only 가 지정되었으나 'celltype' 컬럼이 없습니다.")
                return {"dataset": ds_name, "status": "skipped", "reason": "no celltype column to rename"}
            action = "rename_celltype"
            target_value_desc = "celltype 컬럼 값 유지 및 rename"
        elif cell_type_val:
            action = "assign_new"
            target_value_desc = f"'{cell_type_val}' 할당"
        elif has_celltype:
            # cell_type_val이 지정되지 않았지만 celltype 컬럼이 있는 경우 기본적으로 복사/rename
            action = "rename_celltype"
            target_value_desc = "celltype 컬럼 값 유지 및 복사"
        else:
            log.warning(f"  [{ds_name}] 주입할 --cell-type 값이 지정되지 않았고 기존 'celltype' 컬럼도 없습니다.")
            return {"dataset": ds_name, "status": "skipped", "reason": "no cell_type value given"}

    log.info(f"  적용할 작업: {action} ({target_value_desc})")

    if dry_run:
        log.info(f"  [DRY-RUN] 실제 파일 저장을 생략합니다.")
        return {"dataset": ds_name, "status": "dry_run", "action": action, "value": target_value_desc}

    # h5py로 obs 업데이트
    tmp_path = h5ad_path + ".ct_tmp"
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as h5f:
            obs = h5f["obs"]

            if action == "assign_new":
                # 기존 cell_type 삭제 (있으면)
                if TARGET_COL in obs:
                    del obs[TARGET_COL]

                # 단일 카테고리 categorical로 생성 (메모리/디스크 초절약)
                cat_grp = obs.create_group(TARGET_COL)
                cat_grp.attrs["encoding-type"]    = "categorical"
                cat_grp.attrs["encoding-version"] = "0.2.0"
                cat_grp.attrs["ordered"]          = False

                dt = h5py.string_dtype(encoding="utf-8")
                cat_grp.create_dataset("categories", data=np.array([cell_type_val], dtype=object), dtype=dt)
                # 모든 셀의 코드가 0
                codes = np.zeros(n_cells, dtype=np.int8)
                cat_grp.create_dataset("codes", data=codes)

                update_column_order(obs, TARGET_COL, old_col="celltype" if has_celltype else None)

            elif action == "rename_celltype":
                # 기존 celltype 복사하여 cell_type 생성
                src_ds = obs["celltype"]
                if TARGET_COL in obs:
                    del obs[TARGET_COL]

                if isinstance(src_ds, h5py.Group) and "categories" in src_ds:
                    cat_grp = obs.create_group(TARGET_COL)
                    for k, v in src_ds.attrs.items():
                        cat_grp.attrs[k] = v
                    cat_grp.attrs["encoding-type"]    = "categorical"
                    cat_grp.attrs["encoding-version"] = "0.2.0"
                    cat_grp.attrs["ordered"]          = False

                    cat_grp.create_dataset("categories", data=src_ds["categories"][:], dtype=src_ds["categories"].dtype)
                    cat_grp.create_dataset("codes", data=src_ds["codes"][:], dtype=src_ds["codes"].dtype)
                else:
                    obs.create_dataset(TARGET_COL, data=src_ds[:], dtype=src_ds.dtype)
                    obs[TARGET_COL].attrs["encoding-type"] = "string-array"
                    obs[TARGET_COL].attrs["encoding-version"] = "0.2.0"

                update_column_order(obs, TARGET_COL, old_col="celltype")

        shutil.move(tmp_path, h5ad_path)
        log.info(f"  [{ds_name}] 저장 완료 -> {h5ad_path}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {"dataset": ds_name, "status": "done", "action": action, "value": target_value_desc}


def parse_args():
    p = argparse.ArgumentParser(description="AnnData obs cell_type 표준화 및 추가")
    p.add_argument("--dataset", default=None, help="대상 데이터셋 이름 (e.g., orion_hct116, nadig_hepg2)")
    p.add_argument("--cell-type", default=None, help="설정할 cell_type 이름 (e.g., hct116, hepg2)")
    p.add_argument("--rename-only", action="store_true", help="기존 'celltype' 컬럼을 'cell_type'으로 복사/rename만 수행")
    p.add_argument("--overwrite", action="store_true", help="기존 'cell_type' 컬럼 덮어쓰기")
    p.add_argument("--all", action="store_true", help="기본 프리셋 매핑 테이블을 사용하여 전체 데이터셋 일괄 처리")
    p.add_argument("--dry-run", action="store_true", help="실제 파일 수정 없이 변경 사항만 확인")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="standardized h5ad 파일 경로")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.all and not args.dataset:
        log.error("오류: --dataset [이름] 또는 --all 옵션을 지정해야 합니다. (--help 참조)")
        return

    results = []

    if args.all:
        log.info("전체 데이터셋 기본 프리셋 처리 모드 (--all)")
        files = sorted(glob.glob(os.path.join(args.output_dir, "*_standardized.h5ad")))
        for f in files:
            ds_name = os.path.basename(f).replace("_standardized.h5ad", "")
            preset_val = DEFAULT_CELL_TYPES.get(ds_name, None)

            # 프리셋에 없거나 mixscale/arc_h1 처럼 이미 고유한 cell_type이 있는 경우
            if preset_val is None:
                if ds_name == "kolf_strong":
                    # celltype rename
                    r = process_h5ad(f, ds_name, cell_type_val=None, rename_only=True,
                                     overwrite=args.overwrite, dry_run=args.dry_run)
                else:
                    log.info(f"[{ds_name}] 프리셋 없음 또는 건너뜀")
                    r = {"dataset": ds_name, "status": "skipped", "reason": "no preset"}
            else:
                r = process_h5ad(f, ds_name, cell_type_val=preset_val, rename_only=False,
                                 overwrite=args.overwrite, dry_run=args.dry_run)
            results.append(r)
    else:
        # 단일 데이터셋 처리
        clean_name = args.dataset.replace("_standardized.h5ad", "").replace(".h5ad", "")
        h5ad_path = os.path.join(args.output_dir, f"{clean_name}_standardized.h5ad")
        if not os.path.exists(h5ad_path) and os.path.exists(args.dataset):
            h5ad_path = args.dataset

        r = process_h5ad(h5ad_path, clean_name, cell_type_val=args.cell_type,
                         rename_only=args.rename_only, overwrite=args.overwrite,
                         dry_run=args.dry_run)
        results.append(r)

    # 요약 출력
    log.info("\n" + "=" * 65)
    log.info("결과 요약:")
    log.info(f"{'데이터셋':<25} {'상태':<12} {'작업내용'}")
    log.info("-" * 65)
    for r in results:
        status = r.get("status", "?")
        desc = r.get("value") or r.get("reason") or ""
        log.info(f"  {r['dataset']:<23} {status:<12} {desc}")


if __name__ == "__main__":
    main()
