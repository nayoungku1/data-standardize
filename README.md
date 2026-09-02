### Download GENCODE Comprehensive Primary Assembly (PRI) GTF
Put number of version that you looking for in `{VERSION}`.
```bash
mkdir -p gencode
cd gencode
wget "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{VERSION}/gencode.v{VERSION}.primary_assembly.annotation.gtf.gz"
```

### Inspect GENCODE Version of your reference gene symbol file.
```bash
uv run python check_gencode_version.py --version v{VERSION}
```

### Standardize gene symbols
```bash
nohup uv run python -u process_h5ad.py \
  --output-dir /mnt/nas2/projects/vcc-data/standardized \
  --chunk-size 5000 > standardize_all.log 2>&1 &
```

---

## Standardization & Preprocessing Pipeline Scripts

### 1. `verify_standardized.py`
- **용도**: `/mnt/nas2/projects/vcc-data/standardized/` 내 h5ad 파일들의 유전자(`var`)가 GENCODE v32 기준 및 `vcc-2026/gene_names.csv`와 일치하는지, 누락/초과 유전자가 없는지, 순서가 맞는지 전수 검증합니다.
- **사용법**:
  ```bash
  # 전체 standardized 파일 검증
  python verify_standardized.py

  # 특정 디렉토리 지정 검증
  python verify_standardized.py --dir /mnt/nas2/projects/vcc-data/standardized
  ```

---

### 2. `standardize_perturbation_obs.py`
- **용도**: `obs`의 perturbation 컬럼을 `target_gene`으로 통일하고, 타겟 유전자 이름을 GENCODE v32 심볼로 변환하며, 다양한 non-targeting 라벨(`CTRL`, `NTC`, `NT`, `control` 등)을 `non-targeting`으로 표준화합니다.
  - `arc_h1`: 기존 `target_gene` 컬럼 내용 유지 및 표준화
  - `orion_*`: `gene_target` (`CTRL` -> `non-targeting`) -> `target_gene`
  - `kolf_strong`: `gene_target` (`NTC` -> `non-targeting`) -> `target_gene`
  - `mixscale_*`: `gene` (`NT` -> `non-targeting`) -> `target_gene`
  - `replogle_*`, `nadig_*`: `gene` -> `target_gene`
  - `kaggle`: `sgrna_symbol` -> `target_gene`
- **사용법**:
  ```bash
  # 변경 사항 사전 확인 (저장 없음)
  python standardize_perturbation_obs.py --dry-run --skip-gtf

  # 특정 데이터셋만 적용
  python standardize_perturbation_obs.py --dataset orion_hct116

  # 전체 데이터셋 일괄 적용 (기존 컬럼 덮어쓰기)
  python standardize_perturbation_obs.py --overwrite
  ```

---

### 3. `add_ensembl_id_to_var.py`
- **용도**: GENCODE v32 GTF 어노테이션을 기반으로 `var`에 버전 없는 Ensembl ID(`ensembl_id`) 컬럼을 추가합니다. (AnnData `string-array` 규격 준수로 byte string 디코딩 에러 방지)
- **사용법**:
  ```bash
  # 매핑 커버리지 사전 확인 (dry-run)
  python add_ensembl_id_to_var.py --dry-run

  # 특정 데이터셋만 적용
  python add_ensembl_id_to_var.py --dataset nadig_hepg2 --overwrite

  # 전체 데이터셋 일괄 적용
  python add_ensembl_id_to_var.py --overwrite
  ```

---

### 4. `standardize_cell_type_obs.py`
- **용도**: `obs`에 `cell_type` 컬럼이 없는 경우 지정한 세포주 이름(e.g., `HCT116`)을 단일 카테고리로 생성하거나, `celltype` 등으로 명명된 컬럼을 `cell_type`으로 통일합니다.
- **사용법**:
  ```bash
  # 특정 데이터셋에 cell_type 지정 추가
  python standardize_cell_type_obs.py --dataset orion_hct116 --cell-type HCT116
  python standardize_cell_type_obs.py --dataset orion_hek293t --cell-type HEK293T
  python standardize_cell_type_obs.py --dataset nadig_hepg2 --cell-type hepg2

  # kolf_strong의 기존 celltype 컬럼을 cell_type으로 이름만 변경할 때
  python standardize_cell_type_obs.py --dataset kolf_strong --rename-only

  # 기본 프리셋 매핑 테이블로 전체 일괄 적용
  python standardize_cell_type_obs.py --all --dry-run
  python standardize_cell_type_obs.py --all
  ```

---

### 5. `restore_raw_counts.py`
- **용도**: 정규화된 값으로 저장되어 있던 데이터셋의 `X` 행렬을 정수 raw UMI count로 복원/교체합니다.
  - `arc_h1`: $\text{expm1}(X)$ 후 반올림하여 raw integer count 복원 (기존 log1p는 `layers['log1p']`에 백업)
  - `kolf_strong`: `layers['counts']`의 정수 raw count를 `X`로 교체 (기존 z-normalized 행렬은 `layers['normalized']`에 백업)
- **사용법**:
  ```bash
  # 변환 정밀도 및 샘플 사전 확인
  python restore_raw_counts.py --dry-run

  # 특정 데이터셋만 복원
  python restore_raw_counts.py --dataset arc_h1
  python restore_raw_counts.py --dataset kolf_strong

  # 두 데이터셋 모두 복원
  python restore_raw_counts.py
  ```

---

### 6. `extract_ntc_h5ad.py`
- **용도**: 각 standardized h5ad 파일에서 Non-Targeting Control(`target_gene == 'non-targeting'`) 세포들만 슬라이싱하여 `/mnt/nas2/projects/vcc-data/NTC/{dataset}_ntc.h5ad`로 저장합니다.
  - 대용량 데이터셋(50GB+)에서도 메모리 폭발(OOM Killed)을 방지하는 경량 메모리 슬라이싱 적용
  - pandas `StringDtype` 호환성 처리 완료
  - 이미 완료된 파일은 자동 skip하여 중복 작업 방지
- **사용법**:
  ```bash
  # 각 데이터셋별 NTC 세포 수 및 비율 사전 검사
  python extract_ntc_h5ad.py --dry-run

  # 특정 데이터셋만 NTC 추출
  python extract_ntc_h5ad.py --dataset nadig_hepg2

  # 전체 데이터셋 일괄 추출 (이미 완료된 파일은 자동 skip)
  python extract_ntc_h5ad.py

  # 기존 NTC 파일 강제 덮어쓰기
  python extract_ntc_h5ad.py --overwrite
  ```
