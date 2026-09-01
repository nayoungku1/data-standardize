### Download GENCODE Comprehensive Primary Assembly (PRI) GTF
Put number of version that you looking for in `{VERSION}`.
```bash
wget "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_{VERSION}/gencode.v{VERSION}.primary_assembly.annotation.gtf.gz"
```

### Check Gencode Version of your reference file.
```bash
uv run python check_gencode_version.py --version v{VERSION}
```

### Standardize gene symbols
```bash
nohup uv run python -u process_h5ad.py \
  --output-dir /mnt/nas2/projects/vcc-data/standardized \
  --chunk-size 5000 > standardize_all.log 2>&1 &
```
