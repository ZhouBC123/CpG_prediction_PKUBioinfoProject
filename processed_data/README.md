# Processed CpG Datasets

Encoding: A=0, C=1, G=2, T=3, N/other=4.

- `binary/{train,val,test}.npz`: `X` has shape `(samples, 1024)`, `y` has shape `(samples,)`.
- `segmentation/{train,val,test}.npz`: `X` has shape `(samples, 1024)`, `y` has shape `(samples, 1024)`.
- `*_index.tsv` files record chromosome coordinates and sample source.
- Chromosome split is train: all standard chromosomes except chr8/chr9/chr10/chr11, val: chr8/chr9, test: chr10/chr11.
