#!/usr/bin/env bash
# Usage: bash scripts/download_data.sh [ibm|samld|elliptic]
# Requires the Kaggle CLI and a token in ~/.kaggle/kaggle.json (never commit it).
set -euo pipefail
case "${1:-ibm}" in
  ibm)
    kaggle datasets download -d ealtman2019/ibm-transactions-for-anti-money-laundering-aml -p data/raw/ibm --unzip ;;
  samld)
    kaggle datasets download -d berkanoztas/synthetic-transaction-monitoring-dataset-aml -p data/raw/samld --unzip ;;
  elliptic)
    kaggle datasets download -d ellipticco/elliptic-data-set -p data/raw/elliptic --unzip ;;
  *) echo "unknown target"; exit 1 ;;
esac
echo "Done. Record file names, row counts and checksums in reports/data_manifest.json."
echo "Elliptic++ (Google Drive) and Elliptic2 (elliptic.co/elliptic2) must be downloaded manually."
