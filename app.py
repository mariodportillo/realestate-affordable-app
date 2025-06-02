import oracledb
import numpy as np
import pandas as pd
import json
from flask import Flask, render_template, request
from utils import evaluate_affordability, find_zip_codes
import os


# --- Oracle connection info ---
wallet_location = "Wallet_AffordApp"
username = os.environ.get("ORACLE_DB_USERNAME")  # Now it won't throw NameError
password = os.environ.get("ORACLE_DB_PASSWORD")  # Do the same for password
wallet_password = os.environ.get("WALLET_PASSWORD")

dsn = "affordapp_high"

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
             config_dir=wallet_location,
             user=username,
             password=password,
             dsn=dsn,
             wallet_location=wallet_location,
             wallet_password=wallet_password
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

                results_chunks = list(chunk_list(results, 10))
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