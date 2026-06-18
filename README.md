# CpG Prediction PKU Bioinformatics Project

Course final project for CpG island prediction.

Selected topic: implement neural-network-based CpG island recognition and compare it with probabilistic baselines.

Planned tasks:

1. Binary classification: decide whether a DNA sequence/window is a CpG island.
2. Base-level segmentation: predict whether each base belongs to a CpG island.

Planned model families:

- Markov chain / Hidden Markov Model baselines
- Fully connected neural network
- Convolutional neural network

## Environment

Create the conda environment:

```bash
conda create -n cpg-prediction python=3.10 pip -y
conda activate cpg-prediction
python -m pip install -r requirements.txt
python -m ipykernel install --user --name cpg-prediction --display-name "Python (cpg-prediction)"
```

`environment.yml` records the same environment entry point for conda-based reproduction. The environment uses conda for isolation and pip for project packages, including PyTorch. On GPU machines, verify CUDA availability after creation with:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## Raw Data

Raw data is stored outside the git repository:

```text
/root/autodl-tmp/bioinfo_data/raw/ucsc_hg38
```

Download UCSC hg38 CpG island annotations and reference genome:

```bash
bash scripts/download_ucsc_hg38_cpg.sh
```

See [docs/data_sources.md](docs/data_sources.md) for source URLs and file usage.
