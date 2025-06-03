import oracledb
import pandas as pd
import json
import os

wallet_location = "Wallet_AffordApp"
username = os.environ.get("ORACLE_DB_USERNAME")
password = os.environ.get("ORACLE_DB_PASSWORD")
wallet_password = os.environ.get("WALLET_PASSWORD")
dsn = "affordapp_high"


def load_json_file(filepath):
    """Helper to load existing JSON data or return empty dict if file not found."""
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load JSON from {filepath}: {e}")
    return {}


def extract_means_by_zip(output_path="data/zip_mean_prices.json"):
    try:
        os.environ['TNS_ADMIN'] = wallet_location

        conn = oracledb.connect(
            user=username,
            password=password,
            dsn=dsn,
            wallet_location=wallet_location,
            wallet_password=wallet_password
        )
    except Exception as e:
        print("Failed to connect to Oracle DB:", str(e))
        return pd.DataFrame()

    try:
        query = """
            SELECT zip_code, list_price
            FROM listings
            WHERE zip_code IS NOT NULL AND list_price IS NOT NULL
        """

        df = pd.read_sql(query, conn)
    except Exception as e:
        print("Failed to fetch data:", str(e))
        conn.close()
        return pd.DataFrame()

    conn.close()

    df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce").astype("float64")
    df = df.dropna(subset=["list_price"])

    new_means = (
        df.groupby("zip_code")["list_price"]
        .mean()
        .round(2)
        .to_dict()
    )
    new_means = {str(int(float(k))).zfill(5): v for k, v in new_means.items()}

    # Load existing data and update
    existing_data = load_json_file(output_path)
    # Update existing dict with new means (new keys added or existing keys updated)
    existing_data.update(new_means)

    # Write merged data back
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

    print(f"[INFO] Saved mean prices for {len(new_means)} ZIP codes (merged) to {output_path}")


def generate_zip_metadata(metadata_path="data/zip_metadata.json"):
    try:
        os.environ['TNS_ADMIN'] = wallet_location

        conn = oracledb.connect(
            user=username,
            password=password,
            dsn=dsn,
            wallet_location=wallet_location,
            wallet_password=wallet_password
        )
    except Exception as e:
        print("Failed to connect to Oracle DB:", str(e))
        return

    try:
        query = """
            SELECT DISTINCT zip_code AS zip, city, state
            FROM listings
            WHERE zip_code IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL
        """

        df = pd.read_sql(query, conn)
    except Exception as e:
        print("Failed to fetch data:", str(e))
        conn.close()
        return

    conn.close()

    new_metadata = {
        str(int(float(row["zip"]))).zfill(5): {
            "city": row["city"],
            "state": row["state"]
        }
        for _, row in df.iterrows()
    }

    # Load existing metadata and merge without duplicates
    existing_metadata = load_json_file(metadata_path)
    # Update existing metadata with new entries (overwrites if zip exists)
    existing_metadata.update(new_metadata)

    with open(metadata_path, "w") as f:
        json.dump(existing_metadata, f, indent=2)

    print(f"[INFO] Metadata saved to {metadata_path} (merged)")


if __name__ == "__main__":
    extract_means_by_zip()
    generate_zip_metadata()
