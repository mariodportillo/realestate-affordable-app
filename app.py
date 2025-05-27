import sqlite3
import numpy as np
import pandas as pd
import json
from flask import Flask, render_template, request
from utils import evaluate_affordability

app = Flask(__name__)

thetas = np.load("model/weights.npy")
with open("model/stats.json", "r") as f:
    norm_stats = json.load(f)


def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def load_listings_for_zip(zipcode):
    conn = sqlite3.connect("data/listings.db")
    query = "SELECT * FROM listings WHERE zip_code = ?"
    df = pd.read_sql_query(query, conn, params=(zipcode,))
    conn.close()
    return df


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            user_inputs = {
                "grossIncome": float(request.form["income"]),
                "monthly_debts": float(request.form["debts"]),
                "down_payment": float(request.form["down"]),
                "annual_interest_rate": float(request.form["rate"]),
                "loan_term_in_years": int(request.form["term"]),
            }
            zipcode = int(request.form["zipcode"])

            listings_df = load_listings_for_zip(zipcode)

            results = evaluate_affordability(zipcode, listings_df, user_inputs, thetas, norm_stats)

            results_chunks = list(chunk_list(results, 10))

            return render_template("index.html", results_chunks=results_chunks)
        except Exception as e:
            return render_template("index.html", error=str(e))

    return render_template("index.html")


if __name__ == "__main__":
    app.run()
