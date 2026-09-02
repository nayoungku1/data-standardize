"""
restore_raw_counts.py
---------------------
standardized 디렉토리의 데이터셋 중 raw count가 아닌 arc_h1과 kolf_strong의 X를
raw count로 복원/교체하는 스크립트.

1. kolf_strong:
   - 원본/standardized 파일의 layers['counts']에 정수 raw count(CSR)가 존재함
   - X를 layers['counts']로 교체하고, 기존 normalized X는 layers['normalized']로 보존

2. arc_h1:
   - X가 log1p(raw_count)로 저장되어 있음 (expm1(data)를 취하면 정확한 정수 UMI count 복원)
   - CSR의 sparsity 구조(0의 expm1은 0)는 그대로 유지되므로, data 배열을 expm1 후 반올림
   - 기존 log1p X는 layers['log1p']로 백업 보존

Usage:
    # 1. 시뮬레이션 및 데이터 샘플 미리보기
    python restore_raw_counts.py --dry-run

    # 2. 특정 데이터셋만 실행
    python restore_raw_counts.py --dataset arc_h1
    python restore_raw_counts.py --dataset kolf_strong

    # 3. 두 데이터셋 모두 실행
    python restore_raw_counts.py
"""

import argparse
import logging
import os
import shutil
import time

import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

STANDARDIZED_DIR = "/mnt/nas2/projects/vcc-data/standardized"


def restore_arc_h1(h5ad_path: str, dry_run: bool = False, backup_layer: bool = True) -> dict:
    """arc_h1의 X (log1p CSR)를 expm1을 통해 raw integer count CSR로 변환"""
    log.info(f"\n[arc_h1] 검사 및 변환: {h5ad_path}")

    with h5py.File(h5ad_path, "r") as f:
        if "X" not in f or not isinstance(f["X"], h5py.Group) or "data" not in f["X"]:
            log.error("  [arc_h1] X CSR 그룹을 찾을 수 없습니다.")
            return {"dataset": "arc_h1", "status": "error", "reason": "invalid X structure"}

        sample_d = f["X"]["data"][:20]
        expm1_sample = np.expm1(sample_d)
        round_sample = np.round(expm1_sample)
        max_diff = float(np.max(np.abs(expm1_sample - round_sample)))

        log.info(f"  현재 X data 샘플 (log1p): {sample_d[:5].tolist()}")
        log.info(f"  복원될 raw counts 샘플  : {round_sample[:5].tolist()}")
        log.info(f"  정수 반올림 최대 오차     : {max_diff:.2e} (부동소수점 정밀도 일치)")

        n_elements = len(f["X"]["data"])
        log.info(f"  총 non-zero 요소 수: {n_elements:,}개")

    if dry_run:
        log.info("  [DRY-RUN] 실제 수정을 생략합니다.")
        return {"dataset": "arc_h1", "status": "dry_run", "max_diff": max_diff}

    t0 = time.time()
    tmp_path = h5ad_path + ".raw_tmp"
    log.info(f"  임시 파일 복사 중: {tmp_path}")
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as f:
            x_grp = f["X"]

            # 1. 기존 log1p X 백업
            if backup_layer:
                if "layers" not in f:
                    f.create_group("layers")
                layers = f["layers"]
                if "log1p" in layers:
                    del layers["log1p"]

                log1p_layer = layers.create_group("log1p")
                for k, v in x_grp.attrs.items():
                    log1p_layer.attrs[k] = v

                for ds_name in ["data", "indices", "indptr"]:
                    log1p_layer.create_dataset(ds_name, data=x_grp[ds_name][:])
                log.info("  기존 log1p X를 layers['log1p']에 백업 완료")

            # 2. X['data']를 청크 단위로 expm1 및 round 적용하여 in-place 갱신
            log.info("  X['data'] expm1 변환 및 정수화 진행 중...")
            data_ds = x_grp["data"]
            chunk_size = 5_000_000

            for i in range(0, n_elements, chunk_size):
                end = min(i + chunk_size, n_elements)
                chunk = data_ds[i:end]
                restored = np.round(np.expm1(chunk)).astype(np.float32)
                data_ds[i:end] = restored

            # 속성 업데이트 (encoding-type 등 유지 확인)
            x_grp.attrs["is_raw"] = True

        shutil.move(tmp_path, h5ad_path)
        elapsed = time.time() - t0
        log.info(f"  [arc_h1] raw count 복원 완료! ({elapsed:.1f}초)")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {"dataset": "arc_h1", "status": "done", "elapsed_sec": round(elapsed, 1)}


def restore_kolf_strong(h5ad_path: str, dry_run: bool = False, backup_layer: bool = True) -> dict:
    """kolf_strong의 layers['counts'] (raw UMI counts)로 X를 교체"""
    log.info(f"\n[kolf_strong] 검사 및 변환: {h5ad_path}")

    with h5py.File(h5ad_path, "r") as f:
        if "layers" not in f or "counts" not in f["layers"]:
            log.error("  [kolf_strong] layers['counts']를 찾을 수 없습니다.")
            return {"dataset": "kolf_strong", "status": "error", "reason": "no layers['counts']"}

        counts_grp = f["layers"]["counts"]
        sample_counts = counts_grp["data"][:20]
        is_int = bool(np.all(sample_counts == np.round(sample_counts)))

        log.info(f"  layers['counts'] 샘플: {sample_counts[:5].tolist()}")
        log.info(f"  카운트 정수 여부      : {is_int}")

        if "X" in f and isinstance(f["X"], h5py.Group) and "data" in f["X"]:
            sample_curr_x = f["X"]["data"][:5]
            log.info(f"  현재 X data 샘플      : {sample_curr_x.tolist()} (z-normalized)")

        n_elements = len(counts_grp["data"])
        log.info(f"  총 non-zero 요소 수: {n_elements:,}개")

    if dry_run:
        log.info("  [DRY-RUN] 실제 수정을 생략합니다.")
        return {"dataset": "kolf_strong", "status": "dry_run", "is_int": is_int}

    t0 = time.time()
    tmp_path = h5ad_path + ".raw_tmp"
    log.info(f"  임시 파일 복사 중 (21GB): {tmp_path}")
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as f:
            # 1. 기존 X를 layers['normalized']로 백업
            if backup_layer:
                layers = f["layers"]
                if "normalized" in layers:
                    del layers["normalized"]

                norm_layer = layers.create_group("normalized")
                x_grp = f["X"]
                for k, v in x_grp.attrs.items():
                    norm_layer.attrs[k] = v

                for ds_name in ["data", "indices", "indptr"]:
                    norm_layer.create_dataset(ds_name, data=x_grp[ds_name][:])
                log.info("  기존 X를 layers['normalized']에 백업 완료")

            # 2. X 삭제 후 layers['counts']로 재작성
            del f["X"]
            new_x = f.create_group("X")
            counts_grp = f["layers"]["counts"]

            for k, v in counts_grp.attrs.items():
                new_x.attrs[k] = v

            for ds_name in ["data", "indices", "indptr"]:
                new_x.create_dataset(
                    ds_name,
                    data=counts_grp[ds_name][:],
                    compression="gzip",
                    compression_opts=4
                )

            new_x.attrs["is_raw"] = True
            log.info("  layers['counts']를 X로 복사/교체 완료")

        shutil.move(tmp_path, h5ad_path)
        elapsed = time.time() - t0
        log.info(f"  [kolf_strong] raw count 복원 완료! ({elapsed:.1f}초)")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {"dataset": "kolf_strong", "status": "done", "elapsed_sec": round(elapsed, 1)}


def parse_args():
    p = argparse.ArgumentParser(description="arc_h1 및 kolf_strong의 X를 raw count로 복원/교체")
    p.add_argument("--dataset", choices=["arc_h1", "kolf_strong", "all"], default="all",
                   help="처리할 데이터셋 (arc_h1, kolf_strong, all)")
    p.add_argument("--input-dir", default=STANDARDIZED_DIR, help="표준화 디렉토리 경로")
    p.add_argument("--no-backup", action="store_true", help="기존 normalized/log1p layer 백업 생략")
    p.add_argument("--dry-run", action="store_true", help="실제 변경 없이 데이터 샘플 및 정밀도 확인")
    return p.parse_args()


def main():
    args = parse_args()
    backup_layer = not args.no_backup

    targets = ["arc_h1", "kolf_strong"] if args.dataset == "all" else [args.dataset]
    log.info(f"Raw Count 복원 작업 시작: {targets}")
    log.info(f"디렉토리: {args.input_dir}")
    log.info(f"Dry-run 여부: {args.dry_run}, Layer 백업: {backup_layer}")

    results = []
    for ds in targets:
        h5ad_path = os.path.join(args.input_dir, f"{ds}_standardized.h5ad")
        if not os.path.exists(h5ad_path):
            log.error(f"파일이 존재하지 않습니다: {h5ad_path}")
            results.append({"dataset": ds, "status": "missing"})
            continue

        try:
            if ds == "arc_h1":
                r = restore_arc_h1(h5ad_path, dry_run=args.dry_run, backup_layer=backup_layer)
            elif ds == "kolf_strong":
                r = restore_kolf_strong(h5ad_path, dry_run=args.dry_run, backup_layer=backup_layer)
            results.append(r)
        except Exception as e:
            log.error(f"[{ds}] 처리 중 오류 발생: {e}", exc_info=True)
            results.append({"dataset": ds, "status": "error", "reason": str(e)})

    log.info("\n" + "=" * 55)
    log.info("결과 요약:")
    log.info(f"{'데이터셋':<20} {'상태':<12} {'소요시간/결과'}")
    log.info("-" * 55)
    for r in results:
        status = r.get("status", "?")
        desc = f"{r.get('elapsed_sec', 0)}초" if "elapsed_sec" in r else r.get("reason", "")
        log.info(f"  {r['dataset']:<18} {status:<12} {desc}")


if __name__ == "__main__":
    main()
