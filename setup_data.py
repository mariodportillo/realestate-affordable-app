import os
import oracledb

wallet_location = os.getenv("ORACLE_WALLET_LOCATION") or "/Users/marioportillo/Documents/cs109/cs109_project/Wallet_AffordApp"
username = os.getenv("ORACLE_DB_USERNAME")
password = os.getenv("ORACLE_DB_PASSWORD")
dsn = "affordapp_high"

if not all([wallet_location, username, password, dsn]):
    missing = [name for name, val in {
        "ORACLE_WALLET_LOCATION": wallet_location,
        "ORACLE_DB_USERNAME": username,
        "ORACLE_DB_PASSWORD": password,
        "ORACLE_DB_DSN": dsn
    }.items() if not val]
    raise ValueError(f"Missing environment variables: {missing}")

oracledb.init_oracle_client(
    lib_dir="/Users/marioportillo/oracle/instantclient/instantclient-basic-macos.arm64-23.3.0.23.09-2/",
    config_dir=wallet_location
)

# Constants: update these
CSV_PATH = "/Users/marioportillo/Documents/cs109/cs109_project/Backup_data/EVERY_LISTING_backup.csv"        # your CSV file path
TABLE_NAME = "LISTINGS"                # your Oracle table name (usually uppercase)
DATE_COLUMNS = ["LISTING_DATE"]       # example date columns in Oracle table, uppercase!
UNIQUE_KEYS = ["FULL_STREET_LINE", "MLS_ID", "PROPERTY_URL"]                   # example unique key columns, uppercase!

def parse_date(value):
    if value is None or value == '':
        return None
    try:
        # Adjust format string to your CSV date format
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None

def row_exists(cursor, row):
    """Check if a row with same unique key values exists."""
    where_clause = " AND ".join([f"{key} = :{key}" for key in UNIQUE_KEYS])
    query = f"SELECT 1 FROM {TABLE_NAME} WHERE {where_clause} FETCH FIRST 1 ROWS ONLY"
    bind_vars = {key: row[key] for key in UNIQUE_KEYS}
    cursor.execute(query, bind_vars)
    return cursor.fetchone() is not None

def insert_row(cursor, columns, row):
    """Insert one row into the Oracle table, converting dates properly."""
    cols_str = ", ".join(columns)
    placeholders = []
    bind_vars = {}

    for col in columns:
        val = row.get(col)
        if col in DATE_COLUMNS and val is not None:
            # Oracle date from Python datetime
            placeholders.append(f":{col}")
            bind_vars[col] = val
        else:
            placeholders.append(f":{col}")
            bind_vars[col] = val

    placeholders_str = ", ".join(placeholders)
    query = f"INSERT INTO {TABLE_NAME} ({cols_str}) VALUES ({placeholders_str})"
    cursor.execute(query, bind_vars)

def main(csv_path, table_name):
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = conn.cursor()

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        columns = [col.upper() for col in reader.fieldnames]

        count_inserted = 0
        for row in reader:
            # Normalize keys and parse dates
            row = {k.upper(): v for k, v in row.items()}

            for date_col in DATE_COLUMNS:
                if date_col in row:
                    row[date_col] = parse_date(row[date_col])

            # Check for unique keys presence
            if not all(key in row for key in UNIQUE_KEYS):
                print(f"Skipping row due to missing unique keys: {row}")
                continue

            if not row_exists(cursor, row):
                insert_row(cursor, columns, row)
                count_inserted += 1

        conn.commit()
        print(f"Inserted {count_inserted} new unique rows into {table_name}.")

    cursor.close()
    conn.close()

if __name__ == "__main__":
    main(CSV_PATH, TABLE_NAME)
