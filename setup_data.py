import pandas as pd
import sqlite3
import gzip
import pyarrow.feather as feather
import io

# Load compressed Feather file
with gzip.open("data/EVERY_LISTING.gz", "rb") as f:
    df = feather.read_feather(io.BytesIO(f.read()))

# Connect to (or create) SQLite database file
conn = sqlite3.connect("data/listings.db")

# Write dataframe to SQLite table named 'listings'
df.to_sql("listings", conn, if_exists="replace", index=False)

conn.close()
print("Database created successfully!")
