"""
add_ensembl_id_to_var.py
------------------------
각 standardized h5ad 파일의 var에 'ensembl_id' 컬럼을 추가.

var의 index (gene_name, GENCODE v32 symbol)를 기준으로
GENCODE v32 GTF에서 Ensembl ID를 조회하여 추가.

- 매핑 불가한 gene은 '' (빈 문자열) 처리
- 기존 var 컬럼은 모두 유지
- h5py로 var 그룹에 직접 dataset 추가 (X/layers 불변)
- --dry-run: 실제 저장 없이 커버리지 확인
- --overwrite: 기존 ensembl_id 컬럼 덮어쓰기

Usage:
    python add_ensembl_id_to_var.py
    python add_ensembl_id_to_var.py --dry-run
    python add_ensembl_id_to_var.py --dataset nadig_hepg2
    python add_ensembl_id_to_var.py --overwrite
"""

import argparse
import gzip
import json
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

# ============================================================
# 설정
# ============================================================

META_JSON  = "/mnt/nas2/projects/vcc-data/datasets_meta.json"
GTF_GZ     = "/home/dev02/integration/gencode/gencode.v32.primary_assembly.annotation.gtf.gz"
OUTPUT_DIR = "/mnt/nas2/projects/vcc-data/standardized"


# ============================================================
# 1. GENCODE v32 → gene_name: ensembl_id 매핑 구축
# ============================================================

def load_v32_name_to_ensembl(gtf_gz: str) -> dict:
    """
    GENCODE v32 GTF에서 gene_name → ensembl_id (버전 없는 ENSG...) 매핑.
    동일 gene_name에 여러 Ensembl ID가 있을 경우 첫 번째 등장값 사용.
    """
    log.info(f"GENCODE v32 GTF 로딩: {gtf_gz}")
    name2ens = {}
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
            if gene_name and gene_id:
                ensg = gene_id.split(".")[0]  # 버전 제거: ENSG00000123456.7 → ENSG00000123456
                if gene_name not in name2ens:
                    name2ens[gene_name] = ensg

    log.info(f"  매핑 완료: {len(name2ens):,} gene symbols → Ensembl IDs")
    return name2ens


# ============================================================
# 2. h5py var 구조 확인 헬퍼
# ============================================================

def get_var_gene_names(h5f: h5py.File) -> list:
    """var의 index (gene_name 순서)를 반환"""
    var = h5f["var"]
    # anndata는 var의 index를 '_index' 키 또는 var.attrs['_index']로 저장
    if "_index" in var:
        ds = var["_index"]
        return [v.decode() if isinstance(v, bytes) else str(v) for v in ds[:]]
    # 혹은 'gene_name' 컬럼이 index인 경우
    idx_name = var.attrs.get("_index", None)
    if idx_name and idx_name in var:
        ds = var[idx_name]
        if isinstance(ds, h5py.Group):
            # categorical
            codes = ds["codes"][:]
            cats  = [c.decode() if isinstance(c, bytes) else str(c) for c in ds["categories"][:]]
            return [cats[c] for c in codes]
        else:
            return [v.decode() if isinstance(v, bytes) else str(v) for v in ds[:]]
    raise ValueError(f"var에서 gene index를 찾을 수 없음. keys={list(var.keys())}, attrs={dict(var.attrs)}")


# ============================================================
# 3. 단일 데이터셋 처리
# ============================================================

def process_dataset(ds_name: str, h5ad_path: str,
                    name2ens: dict,
                    dry_run: bool = False,
                    overwrite: bool = False) -> dict:
    log.info(f"\n[{ds_name}] 처리: {h5ad_path}")

    with h5py.File(h5ad_path, "r") as h5f:
        # 기존 ensembl_id 컬럼 확인
        if "ensembl_id" in h5f["var"] and not overwrite:
            log.info(f"  [{ds_name}] var에 'ensembl_id' 이미 존재. --overwrite로 덮어쓰기 가능.")
            return {"dataset": ds_name, "status": "skipped", "reason": "already exists"}

        gene_names = get_var_gene_names(h5f)

    n_genes = len(gene_names)

    # gene_name → ensembl_id 매핑
    ensembl_ids = [name2ens.get(g, "") for g in gene_names]

    n_mapped   = sum(1 for e in ensembl_ids if e)
    n_unmapped = n_genes - n_mapped
    coverage   = n_mapped / n_genes * 100 if n_genes > 0 else 0

    log.info(f"  genes={n_genes:,}, 매핑 성공={n_mapped:,} ({coverage:.1f}%), 미매핑={n_unmapped:,}")

    if n_unmapped > 0:
        unmapped_genes = [g for g, e in zip(gene_names, ensembl_ids) if not e]
        log.warning(f"  미매핑 gene {n_unmapped}개: {unmapped_genes[:15]}{'...' if n_unmapped > 15 else ''}")

    if dry_run:
        log.info(f"  [DRY-RUN] 저장 생략")
        return {
            "dataset": ds_name, "status": "dry_run",
            "n_genes": n_genes, "n_mapped": n_mapped,
            "n_unmapped": n_unmapped, "coverage_pct": round(coverage, 2),
        }

    # h5py로 var/ensembl_id 추가
    tmp_path = h5ad_path + ".varid_tmp"
    shutil.copyfile(h5ad_path, tmp_path)

    try:
        with h5py.File(tmp_path, "a") as h5f:
            var_grp = h5f["var"]

            # 기존 컬럼 삭제 (overwrite 시)
            if "ensembl_id" in var_grp:
                del var_grp["ensembl_id"]

            # string-array로 저장 (AnnData 0.2.0 string-array 규격 준수)
            dt = h5py.string_dtype(encoding="utf-8")
            ds = var_grp.create_dataset(
                "ensembl_id",
                data=np.array(ensembl_ids, dtype=object),
                dtype=dt,
            )
            ds.attrs["encoding-type"]    = "string-array"
            ds.attrs["encoding-version"] = "0.2.0"

            # column-order 업데이트
            if "column-order" in var_grp.attrs:
                col_order = list(var_grp.attrs["column-order"])
                if "ensembl_id" not in col_order:
                    col_order.append("ensembl_id")
                var_grp.attrs["column-order"] = col_order

        shutil.move(tmp_path, h5ad_path)
        log.info(f"  [{ds_name}] 저장 완료: {h5ad_path}")

    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise e

    return {
        "dataset": ds_name, "status": "done",
        "n_genes": n_genes, "n_mapped": n_mapped,
        "n_unmapped": n_unmapped, "coverage_pct": round(coverage, 2),
    }


# ============================================================
# 4. 메인
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="var에 ensembl_id 추가")
    p.add_argument("--meta-json",  default=META_JSON)
    p.add_argument("--gtf-gz",     default=GTF_GZ)
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    p.add_argument("--dataset",    default=None, help="특정 데이터셋만 처리")
    p.add_argument("--dry-run",    action="store_true", help="저장 없이 커버리지 확인")
    p.add_argument("--overwrite",  action="store_true", help="기존 ensembl_id 컬럼 덮어쓰기")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.meta_json) as f:
        meta = json.load(f)

    name2ens = load_v32_name_to_ensembl(args.gtf_gz)

    targets = {args.dataset: meta[args.dataset]} if args.dataset else meta

    results = []
    for ds_name in targets:
        h5ad_path = os.path.join(args.output_dir, f"{ds_name}_standardized.h5ad")

        if not os.path.exists(h5ad_path):
            log.warning(f"[{ds_name}] 파일 없음: {h5ad_path}")
            results.append({"dataset": ds_name, "status": "missing"})
            continue

        # perttmp 파일이 존재하면 진행 중인 작업이 있으므로 건너뜀
        if os.path.exists(h5ad_path + ".perttmp"):
            log.warning(f"[{ds_name}] .perttmp 파일 존재 (다른 작업 진행 중?), 건너뜀")
            results.append({"dataset": ds_name, "status": "skipped", "reason": ".perttmp exists"})
            continue

        try:
            r = process_dataset(
                ds_name, h5ad_path, name2ens,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
            )
            results.append(r)
        except Exception as e:
            log.error(f"[{ds_name}] ERROR: {e}", exc_info=True)
            results.append({"dataset": ds_name, "status": "error", "reason": str(e)})

    # 결과 요약
    log.info("\n" + "=" * 70)
    log.info("결과 요약:")
    log.info(f"{'데이터셋':<35} {'상태':<12} {'genes':>8} {'매핑':>8} {'미매핑':>8} {'커버리지':>10}")
    log.info("-" * 90)
    for r in results:
        s = r.get("status", "?")
        if s in ("done", "dry_run"):
            log.info(
                f"  {r['dataset']:<35} {s:<12} {r.get('n_genes', 0):>8,} "
                f"{r.get('n_mapped', 0):>8,} {r.get('n_unmapped', 0):>8,} "
                f"{r.get('coverage_pct', 0):>9.1f}%"
            )
        else:
            log.info(f"  {r['dataset']:<35} {s:<12}  {r.get('reason', '')}")

    n_ok = sum(1 for r in results if r.get("status") in ("done", "dry_run"))
    log.info(f"\n처리 완료: {n_ok}/{len(results)}")


if __name__ == "__main__":
    main()
