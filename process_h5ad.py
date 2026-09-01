"""
process_h5ad.py  (v4 - Filtering ONLY overlapping standard genes, No zero padding)
----------------------------------------------------------------------------------
각 h5ad 파일을 VCC 2026 표준 gene 18,533개와 비교하여:
  1. gene symbol을 GENCODE v32로 rename (JSON의 gene_conversion 필드 활용)
  2. standard gene 목록에 존재하는 gene만 filtering (없는 gene은 0-padding 없이 아예 제외)
  3. 겹치는 gene들의 순서를 standard gene_names.csv의 상대적 순서에 맞게 정렬
  4. 최종 h5ad 저장 (Shape: n_cells × n_overlap_genes)

메모리 최적화:
  - IncrementalCSRWriter를 사용하여 청크 단위로 h5py에 직접 기록 (RAM 피크 수백 MB 이하)
  - Dense / CSR / CSC sparse matrix 및 Layer 모두 지원

Usage:
    /home/dev02/miniconda3/envs/prophet_env/bin/python process_h5ad.py --dataset nadig_hepg2 --chunk-size 5000 --overwrite
    /home/dev02/miniconda3/envs/prophet_env/bin/python process_h5ad.py --chunk-size 5000
"""

import argparse
import json
import os
import gc
import time
import logging
import shutil

import numpy as np
import pandas as pd
import scipy.sparse as sp
import h5py
import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================
# 1. 표준 gene 목록 로드
# ============================================================

def load_standard_genes(gene_names_csv: str) -> list:
    df = pd.read_csv(gene_names_csv)
    genes = df.iloc[:, 0].dropna().astype(str).tolist()
    log.info(f"표준 gene 수: {len(genes):,}개")
    return genes


# ============================================================
# 2. gene conversion 딕셔너리 구축
# ============================================================

def build_conversion_dict(meta_entry: dict) -> dict:
    """JSON의 gene_conversion 사용. 없으면 identity."""
    conv = meta_entry.get("gene_conversion", None)
    if conv is not None:
        return conv
    return {g: g for g in meta_entry["gene"]}


# ============================================================
# 3. 인덱스 매핑 (Overlap 유전자만 추출)
# ============================================================

def compute_index_mapping(src_genes: list, conversion: dict, std_genes: list):
    """
    표준 유전자 순서(std_genes)를 기준으로 원본에 존재하는 유전자만 추출.
    (0-padding 없이 오직 겹치는 유전자들만 필터링)

    Returns:
        src_col_indices: 원본 matrix에서 가져올 column 인덱스 (길이 = n_overlap)
        dest_col_indices: 결과 matrix(0 ~ n_overlap-1)에 배치할 column 인덱스
        overlap_genes: 겹치는 유전자 이름 리스트 (표준 순서 유지)
    """
    converted = [conversion.get(g, g) for g in src_genes]
    src_to_idx = {}
    for i, c in enumerate(converted):
        if c not in src_to_idx:
            src_to_idx[c] = i

    src_col = []
    overlap_genes = []
    for sg in std_genes:
        if sg in src_to_idx:
            src_col.append(src_to_idx[sg])
            overlap_genes.append(sg)

    src_col_arr = np.array(src_col, dtype=np.int64)
    dest_col_arr = np.arange(len(src_col), dtype=np.int64)

    return src_col_arr, dest_col_arr, overlap_genes


# ============================================================
# 4. IncrementalCSRWriter: h5py에 CSR을 chunk별로 직접 append
# ============================================================

class IncrementalCSRWriter:
    """
    h5py Group에 CSR sparse matrix를 incremental하게 append.
    전체 matrix를 RAM에 올리지 않아도 됨.
    """

    def __init__(self, h5grp: h5py.Group, n_cols: int, expected_cells: int,
                 dtype=np.float32):
        self.grp        = h5grp
        self.n_cols     = n_cols
        self.dtype      = dtype
        self._nnz       = 0
        self._n_rows    = 0

        # resizable datasets
        self.ds_data    = h5grp.create_dataset(
            "data",    shape=(0,), maxshape=(None,), dtype=dtype,
            chunks=(1 << 20,), compression="gzip", compression_opts=4)
        self.ds_indices = h5grp.create_dataset(
            "indices", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(1 << 20,), compression="gzip", compression_opts=4)
        self.ds_indptr  = h5grp.create_dataset(
            "indptr",  shape=(1,), maxshape=(None,), dtype=np.int64,
            chunks=(1 << 16,), compression="gzip", compression_opts=4)
        self.ds_indptr[0] = 0

        # anndata 호환 attributes
        h5grp.attrs["encoding-type"]    = "csr_matrix"
        h5grp.attrs["encoding-version"] = "0.1.0"
        h5grp.attrs["shape"]            = [expected_cells, n_cols]

    def append_chunk(self, mat: sp.csr_matrix):
        mat = mat.astype(self.dtype)
        mat.sort_indices()
        nnz = mat.nnz
        nr  = mat.shape[0]

        if nnz > 0:
            o = self._nnz
            self.ds_data.resize((o + nnz,))
            self.ds_indices.resize((o + nnz,))
            self.ds_data[o:]    = mat.data
            self.ds_indices[o:] = mat.indices.astype(np.int32)

        op = self._n_rows + 1
        new_iptr = mat.indptr[1:].astype(np.int64) + self._nnz
        self.ds_indptr.resize((op + nr,))
        self.ds_indptr[op: op + nr] = new_iptr

        self._nnz    += nnz
        self._n_rows += nr

    def finalize(self):
        self.grp.attrs["shape"] = [self._n_rows, self.n_cols]
        log.info(f"  Writer 완료: {self._n_rows:,} rows, {self._nnz:,} nnz (cols: {self.n_cols:,})")


# ============================================================
# 5. 내부 처리 함수 (재사용 가능한 핵심 로직)
# ============================================================

def _build_lookup(src_col_indices, dest_col_indices, n_src_hint: int = 0) -> tuple:
    """numpy lookup 배열 생성: lookup[src_idx] = dest_idx (없으면 -1)"""
    n_safe = max(
        int(src_col_indices.max()) + 1 if len(src_col_indices) > 0 else 1,
        n_src_hint
    )
    lookup = np.full(n_safe, -1, dtype=np.int64)
    lookup[src_col_indices] = dest_col_indices
    return lookup, n_safe


def _write_dense(dset, n_cells: int, src_col_indices, dest_col_indices,
                 n_overlap: int, chunk_size: int, writer: IncrementalCSRWriter):
    """Dense matrix에서 overlapping column만 읽어 writer에 append."""
    sort_order  = np.argsort(src_col_indices)
    sorted_src  = src_col_indices[sort_order]
    sorted_dest = dest_col_indices[sort_order]

    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        n_r = end - start

        # 필요한 overlapping column만 슬라이싱
        sel = dset[start:end, sorted_src]

        out = np.zeros((n_r, n_overlap), dtype=np.float32)
        out[:, sorted_dest] = sel.astype(np.float32)
        del sel

        writer.append_chunk(sp.csr_matrix(out, dtype=np.float32))
        del out
        gc.collect()

        if start % (chunk_size * 20) == 0:
            log.info(f"    Dense chunk: {end:,}/{n_cells:,} (reading {len(sorted_src):,} cols)")


def _write_sparse_csr(grp, n_cells: int, lookup, n_src_safe: int,
                      n_overlap: int, chunk_size: int, writer: IncrementalCSRWriter):
    """CSR sparse matrix에서 overlapping gene만 필터링하여 writer에 append."""
    data_arr    = grp["data"]
    indices_arr = grp["indices"]
    indptr_arr  = grp["indptr"]

    for start in range(0, n_cells, chunk_size):
        end   = min(start + chunk_size, n_cells)
        n_r   = end - start

        iptr       = indptr_arr[start:end + 1].astype(np.int64)
        rs, re     = int(iptr[0]), int(iptr[-1])
        iptr_local = iptr - rs

        c_data = data_arr[rs:re].astype(np.float32)
        c_idx  = indices_arr[rs:re].astype(np.int64)

        in_range = c_idx < n_src_safe
        dest_map = np.where(in_range, lookup[np.minimum(c_idx, n_src_safe - 1)], -1)
        keep     = dest_map >= 0

        if keep.any():
            kd   = c_data[keep]
            ks   = dest_map[keep].astype(np.int32)
            nnzr = np.diff(iptr_local).astype(np.int64)
            rids = np.repeat(np.arange(n_r, dtype=np.int64), nnzr)
            kr   = rids[keep]
            mat  = sp.csr_matrix((kd, (kr, ks)), shape=(n_r, n_overlap), dtype=np.float32)
        else:
            mat  = sp.csr_matrix((n_r, n_overlap), dtype=np.float32)

        writer.append_chunk(mat)
        del c_data, c_idx, iptr_local, mat
        gc.collect()

        if start % (chunk_size * 10) == 0:
            log.info(f"    Sparse chunk: {end:,}/{n_cells:,}")


def _write_sparse_csc(grp, n_cells: int, src_col_indices, dest_col_indices,
                      n_overlap: int, chunk_size: int, writer: IncrementalCSRWriter):
    """CSC sparse matrix에서 overlapping column만 추출하여 CSR writer에 append."""
    data_arr    = grp["data"]
    indices_arr = grp["indices"]   # row 인덱스
    indptr_arr  = grp["indptr"]    # column별 시작 위치
    n_src_cols  = len(indptr_arr) - 1

    all_rows, all_dests, all_vals = [], [], []
    for li, (src_c, dest_c) in enumerate(zip(src_col_indices, dest_col_indices)):
        if src_c >= n_src_cols:
            continue
        cs = int(indptr_arr[src_c])
        ce = int(indptr_arr[src_c + 1])
        if ce <= cs:
            continue
        vals = data_arr[cs:ce].astype(np.float32)
        rows = indices_arr[cs:ce].astype(np.int64)
        all_rows.append(rows)
        all_dests.append(np.full(len(rows), dest_c, dtype=np.int32))
        all_vals.append(vals)
        if li % 2000 == 0:
            log.info(f"    CSC col {li}/{len(src_col_indices)}")

    if all_rows:
        rows_cat = np.concatenate(all_rows)
        cols_cat = np.concatenate(all_dests)
        vals_cat = np.concatenate(all_vals)
        del all_rows, all_dests, all_vals
        gc.collect()

        mat_full = sp.csr_matrix(
            (vals_cat, (rows_cat, cols_cat)),
            shape=(n_cells, n_overlap), dtype=np.float32
        )
        del rows_cat, cols_cat, vals_cat
        gc.collect()

        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            writer.append_chunk(mat_full[start:end])
        del mat_full
        gc.collect()
    else:
        for start in range(0, n_cells, chunk_size):
            end = min(start + chunk_size, n_cells)
            writer.append_chunk(sp.csr_matrix((end - start, n_overlap), dtype=np.float32))


# ============================================================
# 6. X 및 Layer 처리 dispatcher
# ============================================================

def process_X_to_h5(h5src: h5py.File, n_cells: int,
                     src_col_indices, dest_col_indices,
                     n_overlap: int, chunk_size: int,
                     writer: IncrementalCSRWriter,
                     n_src_genes: int = None):
    x = h5src["X"]
    if isinstance(x, h5py.Group):
        enc = x.attrs.get("encoding-type", "csr_matrix")
        n_hint = n_src_genes or 0
        if n_hint > 0:
            lookup, n_safe = _build_lookup(src_col_indices, dest_col_indices, n_hint)
        else:
            hint = int(x["indices"][0:min(100000, len(x["indices"]))].max()) + 1
            lookup, n_safe = _build_lookup(src_col_indices, dest_col_indices, hint)

        if enc == "csr_matrix":
            log.info("  X: CSR sparse")
            _write_sparse_csr(x, n_cells, lookup, n_safe, n_overlap, chunk_size, writer)
        elif enc == "csc_matrix":
            log.info("  X: CSC sparse")
            _write_sparse_csc(x, n_cells, src_col_indices, dest_col_indices,
                              n_overlap, chunk_size, writer)
        else:
            log.warning(f"  X: 알 수 없는 encoding '{enc}'")
    else:
        log.info("  X: dense")
        _write_dense(x, n_cells, src_col_indices, dest_col_indices, n_overlap, chunk_size, writer)


def process_layer_to_h5(h5src: h5py.File, layer_key: str,
                          n_cells: int, src_col_indices, dest_col_indices,
                          n_overlap: int, chunk_size: int,
                          writer: IncrementalCSRWriter,
                          n_src_genes: int = None):
    layer = h5src["layers"][layer_key]
    log.info(f"  Layer '{layer_key}' 처리 중...")

    if isinstance(layer, h5py.Group):
        enc = layer.attrs.get("encoding-type", "")
        if enc == "csr_matrix":
            n_hint = n_src_genes or 0
            hint = max(n_hint, int(layer["indices"][0:min(10000, len(layer["indices"]))].max()) + 1)
            lookup, n_safe = _build_lookup(src_col_indices, dest_col_indices, hint)
            _write_sparse_csr(layer, n_cells, lookup, n_safe, n_overlap, chunk_size, writer)
        elif enc == "csc_matrix":
            _write_sparse_csc(layer, n_cells, src_col_indices, dest_col_indices,
                              n_overlap, chunk_size, writer)
        else:
            log.warning(f"  Layer '{layer_key}': 알 수 없는 encoding '{enc}', 건너뜀")
    else:
        _write_dense(layer, n_cells, src_col_indices, dest_col_indices, n_overlap, chunk_size, writer)


# ============================================================
# 7. var DataFrame 재구성 (Overlap 유전자만 포함)
# ============================================================

def build_new_var(src_var: pd.DataFrame, src_genes: list, conversion: dict, overlap_genes: list) -> pd.DataFrame:
    """겹치는 표준 gene 목록에 대해서만 var DataFrame 생성."""
    converted = [conversion.get(g, g) for g in src_genes]
    rv = src_var.copy()
    rv.index = converted
    rv = rv[~rv.index.duplicated(keep='first')]

    nv = pd.DataFrame(index=pd.Index(overlap_genes, name=src_var.index.name or "gene_name"))
    nv["gene_name"] = overlap_genes

    for col in src_var.columns:
        if col == "gene_name":
            continue
        reindexed_col = rv[col].reindex(overlap_genes)
        if pd.api.types.is_bool_dtype(rv[col].dtype) or rv[col].dtype == object:
            if rv[col].dtype == bool:
                nv[col] = reindexed_col.fillna(False).astype(bool)
            else:
                nv[col] = reindexed_col.fillna("").astype(str)
        elif isinstance(rv[col].dtype, pd.CategoricalDtype):
            nv[col] = reindexed_col.astype(str).fillna("")
        else:
            nv[col] = reindexed_col.values

    return nv


# ============================================================
# 8. 단일 데이터셋 처리 메인 함수
# ============================================================

def process_dataset(name: str, meta_entry: dict, std_genes: list, output_dir: str,
                    chunk_size: int = 2000, overwrite: bool = False) -> dict:
    """
    단일 데이터셋을 표준 gene과 겹치는 gene만 필터링하여 저장.
    Shape: n_cells × n_overlap (0-padding 없음)
    """
    h5ad_path   = meta_entry["h5ad_path"]
    output_path = os.path.join(output_dir, f"{name}_standardized.h5ad")

    if not overwrite and os.path.exists(output_path):
        log.info(f"[{name}] 이미 존재, 건너뜀: {output_path}")
        return {"name": name, "status": "skipped", "output": output_path}

    log.info(f"\n{'='*60}")
    log.info(f"[{name}] 시작: {h5ad_path}")
    t0 = time.time()

    src_genes  = meta_entry["gene"]
    conversion = build_conversion_dict(meta_entry)
    src_col_indices, dest_col_indices, overlap_genes = compute_index_mapping(src_genes, conversion, std_genes)
    n_overlap  = len(overlap_genes)
    log.info(f"[{name}] 원본 {len(src_genes):,}개 중 표준 일치 gene: {n_overlap:,}개 필터링 (0-padding 없음)")

    # obs/var/uns/obsm 로드
    al = ad.read_h5ad(h5ad_path, backed="r")
    obs      = al.obs.copy()
    src_var  = al.var.copy()
    uns      = dict(al.uns)
    obsm     = {k: al.obsm[k] for k in al.obsm.keys()}
    layer_keys = list(al.layers.keys())
    al.file.close()
    del al
    gc.collect()

    n_cells = len(obs)
    log.info(f"[{name}] {n_cells:,} cells, layers: {layer_keys}")

    os.makedirs(output_dir, exist_ok=True)
    tmp_path = output_path + ".tmp"
    new_var  = build_new_var(src_var, src_genes, conversion, overlap_genes)

    # 골격 먼저 저장
    ad.AnnData(obs=obs, var=new_var, uns=uns, obsm=obsm).write_h5ad(tmp_path)
    gc.collect()

    # h5py로 X + layers를 incremental write (Shape: n_cells × n_overlap)
    with h5py.File(h5ad_path, "r") as h5src, h5py.File(tmp_path, "a") as h5dst:

        # X
        if "X" in h5dst:
            del h5dst["X"]
        x_grp   = h5dst.create_group("X")
        x_writer = IncrementalCSRWriter(x_grp, n_cols=n_overlap, expected_cells=n_cells)
        process_X_to_h5(h5src, n_cells, src_col_indices, dest_col_indices,
                         n_overlap, chunk_size, x_writer, n_src_genes=len(src_genes))
        x_writer.finalize()

        # layers
        if layer_keys:
            if "layers" not in h5dst:
                h5dst.create_group("layers")
            lgrp = h5dst["layers"]
            for lk in layer_keys:
                if lk in lgrp:
                    del lgrp[lk]
                lw = IncrementalCSRWriter(lgrp.create_group(lk), n_cols=n_overlap, expected_cells=n_cells)
                process_layer_to_h5(h5src, lk, n_cells, src_col_indices, dest_col_indices,
                                     n_overlap, chunk_size, lw, n_src_genes=len(src_genes))
                lw.finalize()

    shutil.move(tmp_path, output_path)
    elapsed = time.time() - t0
    log.info(f"[{name}] 완료: {output_path} ({n_cells:,}×{n_overlap:,}, {elapsed:.1f}s)")

    return {"name": name, "status": "done", "output": output_path,
            "n_cells": n_cells, "n_overlap_genes": n_overlap, "elapsed_sec": elapsed}


# ============================================================
# 9. JSON gene_conversion 업데이트
# ============================================================

def update_json_with_conversions(json_path: str, gtf_gz_path: str):
    """각 entry에 gene_conversion 필드를 추가."""
    log.info("gene_conversion 계산 시작...")
    from gtfparse import read_gtf
    import mygene

    log.info("GTF 로딩 중...")
    gtf_df   = read_gtf(gtf_gz_path).to_pandas()
    genes_df = gtf_df[gtf_df["feature"] == "gene"].copy()
    genes_df["ensembl_base"] = genes_df["gene_id"].str.split(".").str[0]
    ens2v32  = dict(zip(genes_df["ensembl_base"], genes_df["gene_name"]))
    v32_set  = set(genes_df["gene_name"])
    log.info(f"GTF 완료: {len(v32_set):,} symbols")

    with open(json_path) as f:
        meta = json.load(f)

    mg = mygene.MyGeneInfo()
    for name, entry in meta.items():
        if "gene_conversion" in entry:
            log.info(f"[{name}] 이미 존재")
            continue
        gene_list  = entry["gene"]
        mismatched = [g for g in gene_list if g not in v32_set]
        conv       = {g: g for g in gene_list}
        log.info(f"[{name}] mygene 조회 {len(mismatched):,}개...")
        if mismatched:
            for res in mg.querymany(mismatched, scopes="symbol,alias",
                                    fields="ensembl.gene", species="human", verbose=False):
                q = res["query"]
                if "ensembl" in res:
                    ed   = res["ensembl"]
                    eids = [e.get("gene") for e in ed] if isinstance(ed, list) else [ed.get("gene")]
                    for eid in eids:
                        if eid and eid in ens2v32:
                            conv[q] = ens2v32[eid]
                            break
        entry["gene_conversion"] = conv
        changed = sum(1 for k, v in conv.items() if k != v)
        log.info(f"[{name}] 변환 {changed:,}개")

    try:
        shutil.copyfile(json_path, json_path + ".bak")
        log.info(f"원본 JSON 백업: {json_path}.bak")
    except Exception as e:
        log.warning(f"JSON 백업 생략 (권한 제한): {e}")

    with open(json_path, "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    log.info(f"JSON 저장 완료: {json_path}")


# ============================================================
# 10. CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(description="h5ad VCC 표준 gene 필터링 변환 (0-padding 없음)")
    p.add_argument("--meta-json",  default="/mnt/nas2/projects/vcc-data/datasets_meta.json")
    p.add_argument("--gene-csv",   default="/mnt/nas2/projects/vcc-data/vcc-2026/gene_names.csv")
    p.add_argument("--output-dir", default="/mnt/nas2/projects/vcc-data/standardized")
    p.add_argument("--gtf-gz",
                   default="/home/dev02/integration/gencode/gencode.v32.primary_assembly.annotation.gtf.gz")
    p.add_argument("--dataset",    default=None)
    p.add_argument("--chunk-size", type=int, default=2000,
                   help="row chunk 크기 (RAM 여유 시 5000~10000 권장)")
    p.add_argument("--overwrite",  action="store_true")
    p.add_argument("--update-json-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.meta_json) as f:
        meta = json.load(f)

    std_genes = load_standard_genes(args.gene_csv)

    if args.update_json_only:
        needs = [n for n, e in meta.items() if "gene_conversion" not in e]
        if needs:
            log.info(f"gene_conversion 미계산: {needs}")
            update_json_with_conversions(args.meta_json, args.gtf_gz)
        log.info("--update-json-only 완료")
        return

    if args.dataset is None:
        needs = [n for n, e in meta.items() if "gene_conversion" not in e]
        if needs:
            log.info(f"gene_conversion 미계산 ({len(needs)}개): {needs}")
            log.info("GTF + mygene API로 자동 계산합니다...")
            update_json_with_conversions(args.meta_json, args.gtf_gz)
            with open(args.meta_json) as f:
                meta = json.load(f)

    targets = {args.dataset: meta[args.dataset]} if args.dataset else meta

    results = []
    for name, entry in targets.items():
        try:
            r = process_dataset(name, entry, std_genes,
                                output_dir=args.output_dir,
                                chunk_size=args.chunk_size,
                                overwrite=args.overwrite)
            results.append(r)
        except Exception as e:
            log.error(f"[{name}] ERROR: {e}", exc_info=True)
            results.append({"name": name, "status": "error", "error": str(e)})

    log.info("\n" + "=" * 60 + "\n결과 요약:")
    for r in results:
        s = r["status"]
        if s == "done":
            log.info(f"  OK   {r['name']}: {r['n_cells']:,}×{r['n_overlap_genes']:,} "
                     f"({r['elapsed_sec']:.0f}s)")
        elif s == "skipped":
            log.info(f"  SKIP {r['name']}")
        else:
            log.info(f"  ERR  {r['name']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()
