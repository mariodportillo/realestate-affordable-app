from flask import Flask, render_template, request
import gzip
import pyarrow.feather as feather
import io
import numpy as np
import json
import pandas as pd
from utils import evaluate_affordability

app = Flask(__name__)

# Load model weights and normalization stats once on startup
thetas = np.load("model/weights.npy")
with open("model/stats.json", "r") as f:
    norm_stats = json.load(f)


def chunk_list(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

with gzip.open("data/EVERY_LISTING.gz", "rb") as f:
    # Read Feather from the decompressed bytes
    listings_df = feather.read_feather(io.BytesIO(f.read()))


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Collect user inputs from the form
            user_inputs = {
                "grossIncome": float(request.form["income"]),
                "monthly_debts": float(request.form["debts"]),
                "down_payment": float(request.form["down"]),
                "annual_interest_rate": float(request.form["rate"]),
                "loan_term_in_years": int(request.form["term"]),
            }
            zipcode = int(request.form["zipcode"])

            # Load listings data

            # Get affordability results
            results = evaluate_affordability(zipcode, listings_df, user_inputs, thetas, norm_stats)

            # Chunk results into pages of 10 listings
            results_chunks = list(chunk_list(results, 10))

            return render_template("index.html", results_chunks=results_chunks)
        except Exception as e:
            return render_template("index.html", error=str(e))

    # GET request: just render the page without results
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
