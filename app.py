import oracledb
import numpy as np
import pandas as pd
import ssl
import json
from flask import Flask, render_template, request
from utils import evaluate_affordability, find_zip_codes
import sys
import os


# --- Oracle connection info ---
wallet_location = "Wallet_AffordApp"
username = os.environ.get("ORACLE_DB_USERNAME")  # Now it won't throw NameError
password = os.environ.get("ORACLE_DB_PASSWORD")  # Do the same for password
dsn = "affordapp_high"

os.environ["TNS_ADMIN"] = wallet_location

lib_dir = "/mnt/instantclient/instantclient_21_10"
wallet_dir = "/mnt/wallet/Wallet_AffordApp"

# Check that Oracle Instant Client directory exists and has expected files
if not os.path.isdir(lib_dir):
    print(f"[ERROR] Oracle Instant Client directory does NOT exist: {lib_dir}")
    sys.exit(1)

# Optional: Check that the Instant Client directory contains some expected shared libs
expected_files = ["libclntsh.so", "libocci.so"]
missing_files = [f for f in expected_files if not any(fname.startswith(f) for fname in os.listdir(lib_dir))]
if missing_files:
    print(f"[ERROR] Oracle Instant Client missing expected files: {missing_files}")
    sys.exit(1)

# Check that wallet directory exists and has tnsnames.ora
if not os.path.isdir(wallet_dir):
    print(f"[ERROR] Wallet directory does NOT exist: {wallet_dir}")
    sys.exit(1)

if not os.path.isfile(os.path.join(wallet_dir, "tnsnames.ora")):
    print(f"[ERROR] Wallet missing tnsnames.ora in: {wallet_dir}")
    sys.exit(1)

# If all checks pass, set env and initialize
os.environ["TNS_ADMIN"] = wallet_dir
oracledb.init_oracle_client(lib_dir=lib_dir)

print("[INFO] Oracle client initialized successfully.")



app = Flask(__name__)

thetas = np.load("model/weights.npy")
with open("model/stats.json", "r") as f:
    norm_stats = json.load(f)


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_listings_for_zip(zipcode):
    try:
        os.environ['TNS_ADMIN'] = wallet_location

        conn = oracledb.connect(
            user=username,
            password=password,
            dsn=dsn
        )
    except Exception as e:
        print("Failed to connect", str(e))
        return pd.DataFrame()

    cursor = conn.cursor()

    zipcode = int(zipcode)
    # Updated query to use schema-qualified table name
    query = "SELECT * FROM admin.listings WHERE ZIP_CODE = :zipcode"  # Added admin. prefix
    cursor.execute(query, zipcode=zipcode)  # pass named parameter
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=columns)
    print(f"[DEBUG] Found {len(df)} listings for ZIP {zipcode}")
    return df


@app.route("/", methods=["GET", "POST"])
def index():
    form_values = {}
    if request.method == "POST":
        try:
            action = request.form.get("action")
            form_values = {
                "income": request.form.get("income", ""),
                "debts": request.form.get("debts", ""),
                "down": request.form.get("down", ""),
                "rate": request.form.get("rate", ""),
                "term": request.form.get("term", ""),
                "zipcode": request.form.get("zipcode", ""),
            }

            user_inputs = {
                "grossIncome": float(form_values["income"]),
                "monthly_debts": float(form_values["debts"]),
                "down_payment": float(form_values["down"]),
                "annual_interest_rate": float(form_values["rate"]),
                "loan_term_in_years": int(form_values["term"]),
            }

            if action == "check_affordability":
                zipcode_str = form_values.get("zipcode", "").strip()
                if not zipcode_str.isdigit():
                    raise ValueError("Please enter a valid zipcode (numbers only).")
                zipcode = int(zipcode_str)
                listings_df = load_listings_for_zip(zipcode)

                results = evaluate_affordability(zipcode, listings_df, user_inputs, thetas, norm_stats)
                print(results)

                results_chunks = list(chunk_list(results, 10))

                print(results_chunks)
                affordable_count = sum(1 for r in results if r.get('affordable'))
                not_affordable_count = len(results) - affordable_count

                return render_template(
                    "index.html",
                    results_chunks=results_chunks,
                    form_values=form_values,
                    affordable_count=affordable_count,
                    not_affordable_count=not_affordable_count,
                )

            elif action == "find_affordable_zips":
                affordable_zips = find_zip_codes(user_inputs, json_path="data/zip_mean_prices.json")

                return render_template(
                    "index.html",
                    form_values=form_values,
                    affordable_zips=affordable_zips,
                )

            else:
                return render_template("index.html", form_values=form_values)

        except Exception as e:
            return render_template("index.html", error=str(e), form_values=form_values)

    return render_template("index.html", form_values=form_values)


if __name__ == '__main__':
    app.run()