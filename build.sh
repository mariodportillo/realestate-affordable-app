#!/bin/bash
#!/bin/bash

echo "[INFO] Starting build script..."

# Install Oracle Instant Client if not already available
# (Render uses Ubuntu, so we can use apt or direct download if needed)
# But if you've already included the Instant Client files in your repo, just unpack:
if [ ! -d "instantclient_19_8" ]; then
  echo "[INFO] Unpacking Oracle Instant Client..."
  unzip -q instantclient-basiclite-linux.x64-19.8.0.0.0dbru.zip
fi

# Set up the Oracle environment variables
echo "[INFO] Exporting Oracle environment variables..."
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/instantclient_19_8
export TNS_ADMIN=$(pwd)/Wallet_AffordApp

# Make sure wallet directory exists
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

