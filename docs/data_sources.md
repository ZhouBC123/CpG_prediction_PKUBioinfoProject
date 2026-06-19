# Data Sources

This project uses UCSC Genome Browser hg38 resources as the raw source for both CpG island classification and base-level segmentation. The raw downloads are kept outside git, while the compact processed datasets are tracked in `processed_data/`.

Raw data directory:

```text
/root/autodl-tmp/bioinfo_data/raw/ucsc_hg38
```

Files:

| File | Source | Use |
| --- | --- | --- |
| `cpgIslandExt.txt.gz` | UCSC hg38 database table | CpG island intervals and island-level statistics. |
| `cpgIslandExt.sql` | UCSC hg38 database schema | Column definitions for `cpgIslandExt.txt.gz`. |
| `hg38.fa.gz` | UCSC hg38 bigZips | Reference sequence used to extract positive and negative sequence windows and base-level labels. |
| `hg38.chrom.sizes` | UCSC hg38 bigZips | Chromosome sizes for interval validation and background sampling. |
| `bigZips.md5sum.txt` | UCSC hg38 bigZips | Official checksums for selected bigZips files. |

Download:

```bash
bash scripts/download_ucsc_hg38_cpg.sh
```

The script writes SHA-256 hashes to `sha256sum.txt` and verifies `hg38.fa.gz` plus `hg38.chrom.sizes` against UCSC MD5 checksums.

Preprocessing converts `cpgIslandExt.txt.gz` into interval labels and samples non-CpG windows from `hg38.fa.gz`. The generated datasets are written under the project `processed_data/` directory and are committed to git so the model smoke tests can run without rebuilding hg38 windows.
