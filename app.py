import oracledb
import numpy as np
import pandas as pd
import json
from flask import Flask, render_template, request
from utils import evaluate_affordability, find_zip_codes
import os

# --- Oracle connection info from env vars ---
wallet_location = "Wallet_AffordApp"
username = os.getenv("ORACLE_DB_USERNAME")
password = os.getenv("ORACLE_DB_PASSWORD")
dsn = "affordapp_high"

# Initialize Oracle Instant Client
oracledb.init_oracle_client(lib_dir="/Users/marioportillo/oracle/instantclient/instantclient-basic-macos.arm64-23.3.0.23.09-2/")

app = Flask(__name__)

thetas = np.load("model/weights.npy")
with open("model/stats.json", "r") as f:
    norm_stats = json.load(f)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def load_listings_for_zip(zipcode):
    conn = oracledb.connect(user=username, password=password, dsn=dsn)
    cursor = conn.cursor()

    zipcode = int(zipcode)
    query = "SELECT * FROM listings WHERE zip_code = :zipcode"  # named bind
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
