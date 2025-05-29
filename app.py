import sqlite3
import numpy as np
import pandas as pd
import json
from flask import Flask, render_template, request
from utils import evaluate_affordability, find_zip_codes  # import new function
import os

app = Flask(__name__)

thetas = np.load("model/weights.npy")
with open("model/stats.json", "r") as f:
    norm_stats = json.load(f)

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data/listings.db")

def load_listings_for_zip(zipcode):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT * FROM listings WHERE CAST(zip_code AS INTEGER) = ?"
    df = pd.read_sql_query(query, conn, params=(zipcode,))
    conn.close()
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
                # You may want to pass path to your ZIP price JSON here
                affordable_zips = find_zip_codes(user_inputs, json_path="data/zip_mean_prices.json")

                # affordable_zips is a dict of zip_code (str) : affordability_score (float)
                # Sort descending by score
                return render_template(
                    "index.html",
                    form_values=form_values,
                    affordable_zips=affordable_zips,
                )

            else:
                # Unknown action fallback
                return render_template("index.html", form_values=form_values)

        except Exception as e:
            return render_template("index.html", error=str(e), form_values=form_values)

    return render_template("index.html", form_values=form_values)

if __name__ == '__main__':
    app.run()