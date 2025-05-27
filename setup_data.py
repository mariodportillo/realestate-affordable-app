import pandas as pd
import sqlite3
import gzip
import pyarrow.feather as feather
import io

# conn = sqlite3.connect("data/listings.db")
# df = pd.read_sql_query("SELECT DISTINCT zip_code FROM listings LIMIT 100", conn)
# print(df.head(20))

import cx_Oracle
import subprocess
import os
os.environ["TNS_ADMIN"] = "/Users/marioportillo/Documents/cs109/cs109_project/Wallet_AffordApp"

# Update these variables:
oracle_user = "admin"
oracle_password = os.getenv("ORACLE_DB_PASSWORD")  # or hardcode for testing
oracle_dsn = "affordapp_high"

# This must be the absolute path on the Oracle DB server machine where the dump file is placed
oracle_dir_path = "/path/on/oracle/server/data"  # <-- UPDATE THIS!

# Name of the Oracle DIRECTORY object
oracle_directory_name = "EVER_LISTING_DIR"

# Dump file name inside that directory
dumpfile_name = "EVER_LISTING.dmp"

# Log file name
logfile_name = "import.log"


def create_directory_and_grant():
    connection = cx_Oracle.connect(
        user=oracle_user,
        password=oracle_password,
        dsn=oracle_dsn,
        encoding="UTF-8"
    )
    cursor = connection.cursor()

    # Create or replace directory object
    create_dir_sql = f"""
    CREATE OR REPLACE DIRECTORY {oracle_directory_name} AS '{oracle_dir_path}'
    """
    print(f"[INFO] Creating Oracle DIRECTORY object: {oracle_directory_name} -> {oracle_dir_path}")
    cursor.execute(create_dir_sql)

    # Grant read/write on directory to user
    grant_sql = f"GRANT READ, WRITE ON DIRECTORY {oracle_directory_name} TO {oracle_user}"
    print(f"[INFO] Granting READ, WRITE privileges on directory to user {oracle_user}")
    cursor.execute(grant_sql)

    connection.commit()
    cursor.close()
    connection.close()
    print("[INFO] Directory creation and grants completed.")


def run_impdp():
    # Construct impdp command
    impdp_cmd = [
        "impdp",
        f"{oracle_user}/{oracle_password}@{oracle_dsn}",
        f"DIRECTORY={oracle_directory_name}",
        f"DUMPFILE={dumpfile_name}",
        f"LOGFILE={logfile_name}"
    ]
    print("[INFO] Running impdp command:", " ".join(impdp_cmd))

    # Run the command and stream output
    proc = subprocess.Popen(impdp_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="")
    proc.wait()

    if proc.returncode != 0:
        print(f"[ERROR] impdp failed with exit code {proc.returncode}")
    else:
        print("[INFO] impdp completed successfully.")


if __name__ == "__main__":
    create_directory_and_grant()
    run_impdp()

"""
# Load compressed Feather file
with gzip.open("data/EVERY_LISTING.gz", "rb") as f:
    df = feather.read_feather(io.BytesIO(f.read()))

# Print debug info
print("Loaded DataFrame:")
print(df.head())
print("Columns:", df.columns)
print("Shape:", df.shape)

# Normalize column names (optional)
df.columns = df.columns.str.lower()

# Check for 'zip_code' column
if "zip_code" not in df.columns:
    raise ValueError("Missing required column: 'zip_code'")

# Connect and write to SQLite
conn = sqlite3.connect("data/listings.db")
df.to_sql("listings", conn, if_exists="replace", index=False)
conn.close()

print("Database created successfully!")
"""