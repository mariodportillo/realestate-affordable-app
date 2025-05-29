import sqlite3
import pandas as pd
import json
import os

DB_PATH = "data/listings.db"  # Adjust if your path is different
OUTPUT_PATH = "data/zip_mean_prices.json"

def extract_means_by_zip(db_path, output_path):
    conn = sqlite3.connect(db_path)

    # Load only the columns needed for aggregation
    query = "SELECT zip_code, list_price FROM listings WHERE list_price IS NOT NULL AND zip_code IS NOT NULL"
    df = pd.read_sql_query(query, conn)

    conn.close()

    # Group and compute mean
    df["list_price"] = pd.to_numeric(df["list_price"], errors="coerce").astype("float64")
    df = df.dropna(subset=["list_price"])
    zip_price_means = (
        df.groupby("zip_code")["list_price"]
        .mean()
        .round(2)  # round to 2 decimal places
        .to_dict()
    )

    # Ensure keys are strings padded to 5 digits
    zip_price_means_str_keys = {str(int(float(k))).zfill(5): v for k, v in zip_price_means.items()}

    # Write to JSON
    with open(output_path, "w") as f:
        json.dump(zip_price_means_str_keys, f)

    print(f"[INFO] Saved mean prices for {len(zip_price_means)} ZIP codes to {output_path}")

def generate_zip_metadata(db_path, metadata_path="data/zip_metadata.json"):
    conn = sqlite3.connect(db_path)

    # Load only the columns needed for aggregation
    query = """
        SELECT DISTINCT zip_code AS zip, city, state AS state
        FROM listings
        WHERE zip_code IS NOT NULL AND city IS NOT NULL AND state IS NOT NULL
        """
    df = pd.read_sql_query(query, conn)
    conn.close()

    zip_metadata = {
        str(int(float(row["zip"]))): {
            "city": row["city"],
            "state": row["state"]
        }
        for _, row in df.iterrows()
    }

    # Save to JSON
    with open(metadata_path, "w") as f:
        json.dump(zip_metadata, f, indent=2)

    print(f"Metadata saved to {metadata_path}")

if __name__ == "__main__":
    extract_means_by_zip(DB_PATH, OUTPUT_PATH)
    generate_zip_metadata(DB_PATH)
