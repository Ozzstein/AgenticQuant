#!/usr/bin/env bash
# Download Qlib US stock data to the default provider_uri directory.
set -e

QLIB_DIR="${HOME}/.qlib/qlib_data/us_data"

echo "Downloading Qlib US data to ${QLIB_DIR}..."
conda run -n aiquant python -m qlib.run.get_data qlib_data \
    --target_dir "${QLIB_DIR}" \
    --region us

echo "Verifying download..."
if [ -d "${QLIB_DIR}" ]; then
    echo "Done: ${QLIB_DIR}"
else
    echo "ERROR: Download failed — ${QLIB_DIR} not found."
    exit 1
fi
