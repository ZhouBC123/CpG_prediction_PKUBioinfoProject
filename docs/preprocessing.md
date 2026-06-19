# Preprocessing

The preprocessing step converts UCSC hg38 CpG island annotations and reference genome sequence into two fixed-window datasets under `processed_data/`.

Run:

```bash
conda run -n cpg-prediction python scripts/preprocess_ucsc_cpg.py
```

Outputs:

- `processed_data/binary/{train,val,test}.npz`: window-level binary classification data.
- `processed_data/segmentation/{train,val,test}.npz`: base-level segmentation data.
- `processed_data/*/*_index.tsv`: chromosome coordinates and sample source for each row.
- `processed_data/manifest.json`: generation parameters, split definitions, and output shapes.
- `processed_data/cpg_islands_standard.tsv`: filtered UCSC `cpgIslandExt` rows for standard chromosomes.

Sequence encoding is `A=0`, `C=1`, `G=2`, `T=3`, `N/other=4`.

Chromosome split:

- Train: standard chromosomes except `chr8`, `chr9`, `chr10`, `chr11`
- Validation: `chr8`, `chr9`
- Test: `chr10`, `chr11`

This split keeps windows from the same chromosome in the same partition to reduce genomic interval leakage.
