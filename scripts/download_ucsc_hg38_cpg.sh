#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-${BIOINFO_DATA_DIR:-/root/autodl-tmp/bioinfo_data}}"
RAW_DIR="${DATA_DIR}/raw/ucsc_hg38"
mkdir -p "${RAW_DIR}"

BASE="http://hgdownload.soe.ucsc.edu/goldenPath/hg38"

download() {
  local url="$1"
  local dest="$2"
  if command -v aria2c >/dev/null 2>&1; then
    echo "download/resume: ${url}"
    env -u all_proxy -u ALL_PROXY aria2c --continue=true --max-connection-per-server=8 --split=8 --min-split-size=1M \
      --dir "$(dirname "${dest}")" --out "$(basename "${dest}")" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    echo "download/resume: ${url}"
    wget -c -O "${dest}" "${url}"
  else
    echo "download/resume: ${url}"
    curl -fL --retry 5 --retry-delay 5 --continue-at - -o "${dest}" "${url}"
  fi
}

download "${BASE}/database/cpgIslandExt.txt.gz" "${RAW_DIR}/cpgIslandExt.txt.gz"
download "${BASE}/database/cpgIslandExt.sql" "${RAW_DIR}/cpgIslandExt.sql"
download "${BASE}/bigZips/hg38.fa.gz" "${RAW_DIR}/hg38.fa.gz"
download "${BASE}/bigZips/hg38.chrom.sizes" "${RAW_DIR}/hg38.chrom.sizes"
download "${BASE}/bigZips/md5sum.txt" "${RAW_DIR}/bigZips.md5sum.txt"

(
  cd "${RAW_DIR}"
  grep -E '  (hg38\.fa\.gz|hg38\.chrom\.sizes)$' bigZips.md5sum.txt > md5sum.selected.txt
  md5sum -c md5sum.selected.txt
  sha256sum cpgIslandExt.txt.gz cpgIslandExt.sql hg38.fa.gz hg38.chrom.sizes bigZips.md5sum.txt > sha256sum.txt
)

cat > "${RAW_DIR}/README.txt" <<EOF
UCSC hg38 CpG island project raw data

Downloaded from:
${BASE}/database/cpgIslandExt.txt.gz
${BASE}/database/cpgIslandExt.sql
${BASE}/bigZips/hg38.fa.gz
${BASE}/bigZips/hg38.chrom.sizes
${BASE}/bigZips/md5sum.txt

Purpose:
- cpgIslandExt.txt.gz: CpG island annotation intervals and summary statistics.
- cpgIslandExt.sql: table schema for cpgIslandExt.
- hg38.fa.gz: GRCh38/hg38 reference genome sequence for extracting positive/negative examples and base-level labels.
- hg38.chrom.sizes: chromosome lengths for interval validation and sampling.

Integrity:
- hg38.fa.gz and hg38.chrom.sizes are checked against UCSC bigZips md5sum.txt.
- sha256sum.txt records local SHA-256 hashes for all downloaded files.
EOF

echo "done: ${RAW_DIR}"
