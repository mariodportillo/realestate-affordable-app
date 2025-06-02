#!/bin/bash

echo "[INFO] Starting build script..."

# Define the expected ZIP file and target directory
ZIP_FILE="instantclient-basic-linux.zseries64-19.26.0.0.0dbru.zip"
TARGET_DIR="instantclient_19_26"

# Check if the ZIP file exists in the current directory
if [ ! -f "$ZIP_FILE" ]; then
  echo "[ERROR] Oracle Instant Client ZIP file not found: $ZIP_FILE"
  exit 1
fi

# Unzip the Oracle Instant Client directly into the target directory
if [ ! -d "$TARGET_DIR" ]; then
  echo "[INFO] Unpacking Oracle Instant Client..."
  mkdir -p "$TARGET_DIR"
  unzip -q "$ZIP_FILE" -d "$TARGET_DIR"
fi

# Set up the Oracle environment variables
echo "[INFO] Exporting Oracle environment variables..."
export LD_LIBRARY_PATH="$(pwd)/$TARGET_DIR"
export TNS_ADMIN="$(pwd)/Wallet_AffordApp"

# Verify wallet directory exists
echo "[INFO] Wallet directory is:"
ls Wallet_AffordApp

# Optional: verify wallet files
if [ -f "Wallet_AffordApp/tnsnames.ora" ]; then
  echo "[INFO] Wallet configuration found."
else
  echo "[ERROR] Wallet configuration missing. Please check your wallet files."
  exit 1
fi

echo "[INFO] Build script completed."
