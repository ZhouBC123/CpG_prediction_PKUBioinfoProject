# Preprocessing

The preprocessing step converts UCSC hg38 CpG island annotations and reference genome sequence into two fixed-window datasets under `processed_data/`. The resulting directory is intentionally tracked in git because it is compact, about 40 MB, and removes the need to rebuild data before running model checks.

The repository already includes a generated `processed_data/` snapshot. Rebuild it only after changing preprocessing logic or raw data inputs:

```bash
conda run -n cpg-prediction python scripts/preprocess_ucsc_cpg.py
```

Outputs:

- `processed_data/binary/{train,val,test}.npz`: window-level binary classification data.
- `processed_data/segmentation/{train,val,test}.npz`: base-level segmentation data.
- `processed_data/*/*_index.tsv`: chromosome coordinates and sample source for each row.
- `processed_data/manifest.json`: generation parameters, split definitions, and output shapes.
- `processed_data/cpg_islands_standard.tsv`: filtered UCSC `cpgIslandExt` rows for standard chromosomes.

Dataset sizes in the committed snapshot:

| Task | Split | Samples | Label shape |
| --- | --- | ---: | --- |
| Binary | Train | 46,258 | `(46258,)` |
| Binary | Validation | 4,554 | `(4554,)` |
| Binary | Test | 5,072 | `(5072,)` |
| Segmentation | Train | 46,258 | `(46258, 1024)` |
| Segmentation | Validation | 4,554 | `(4554, 1024)` |
| Segmentation | Test | 5,072 | `(5072, 1024)` |

Sequence encoding is `A=0`, `C=1`, `G=2`, `T=3`, `N/other=4`.

Chromosome split:

- Train: standard chromosomes except `chr8`, `chr9`, `chr10`, `chr11`
- Validation: `chr8`, `chr9`
- Test: `chr10`, `chr11`

This split keeps windows from the same chromosome in the same partition to reduce genomic interval leakage.

Regeneration is deterministic with the default seed recorded in `processed_data/manifest.json`.
