import oracledb
import os

# Use your wallet location
os.environ['TNS_ADMIN'] = 'Wallet_AffordApp'

# Initialize Oracle client
oracledb.init_oracle_client(
    lib_dir="/Users/marioportillo/oracle/instantclient/instantclient-basic-macos.arm64-23.3.0.23.09-2/")

# Test each DSN
dsn_options = [
    "affordapp_high",
    "affordapp_medium",
    "affordapp_low",
    "affordapp_tp",
    "affordapp_tpurgent"
]

username = "webapp_reader"
password = "SecurePassword123!"

for dsn in dsn_options:
    try:
        print(f"Testing connection to {dsn}...")
        conn = oracledb.connect(
            user=username,
            password=password,
            dsn=dsn
        )

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM admin.listings")
        count = cursor.fetchone()[0]
        print(f"✅ {dsn}: Connected successfully! Found {count} listings")
        conn.close()
        break

    except Exception as e:
        print(f"❌ {dsn}: Failed - {str(e)}")

print("Testing complete.")