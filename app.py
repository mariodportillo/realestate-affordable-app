import oracledb
import numpy as np
import pandas as pd
import json
import gzip, io
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, request, render_template_string
from utils import evaluate_affordability, find_zip_codes
from model.training_model import LogisticRegressionRealEstate
import os

IS_RENDER = os.getenv("RENDER") == "true"

if IS_RENDER:
    # --- Oracle connection info ---
    wallet_location = "Wallet_AffordApp"
    username = os.environ.get("ORACLE_DB_USERNAME")  # Env vars for security
    password = os.environ.get("ORACLE_DB_PASSWORD")
    wallet_password = os.environ.get("WALLET_PASSWORD")
else:
    from secret import send_credentials
    cred_dict = send_credentials()
    wallet_location = cred_dict["wallet_location"]
    username = cred_dict["username"]
    password = cred_dict["password"]
    wallet_password = cred_dict["wallet_password"]

dsn = "affordapp_high"

app = Flask(__name__)

thetas = np.load("model/weights.npy")
with open("model/stats.json", "r") as f:
    norm_stats = json.load(f)

with gzip.open("model/ahs2023n.feather.gz", "rb") as f:
    decompressed_bytes = f.read()
df_raw_ahs = pd.read_feather(io.BytesIO(decompressed_bytes))

# Initialize model
default_model = LogisticRegressionRealEstate(learning_rate=0.0005, training_steps=3000)

# Set weights and normalization stats
default_model.thetas = thetas
default_model.norm_stats = norm_stats
default_n = 0.0005
# Normalize using loaded stats (your model.normalize() should use norm_stats)
df_normalized_global = default_model.normalize(df_raw_ahs)

# Train/test split (no retraining here)
train_df_global, test_df_global = train_test_split(df_normalized_global, test_size=0.2, random_state=42)

# Predict and evaluate with stored weights
default_preds = default_model.predict(test_df_global)
default_accuracy = default_model.evaluate(test_df_global, default_preds)

# Extract coefficients for display
default_feature_columns = train_df_global.columns.drop(['Label']).tolist()
default_coefs = {"Bias": default_model.thetas[0]}
default_coefs.update({default_feature_columns[i]: coef for i, coef in enumerate(default_model.thetas[1:])})

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
    query = "SELECT * FROM admin.listings WHERE ZIP_CODE = :zipcode"
    cursor.execute(query, zipcode=zipcode)
    columns = [col[0] for col in cursor.description]
    rows = cursor.fetchall()
    conn.close()
    df = pd.DataFrame(rows, columns=columns)
    return df


@app.route("/train", methods=["GET", "POST"])
def train_model_page():
    result_dict = None
    if request.method == "POST":

        try:
            if IS_RENDER:
                raise Exception("Sorry, can't train the model on RENDER. Need to run locally.")

            n_val = float(request.form.get("n_value", default_n))
            model = LogisticRegressionRealEstate(learning_rate=n_val, training_steps=3000)

            # Normalize and split inside route fresh for each new model
            df_normalized = model.normalize(df_raw_ahs)
            train_df, test_df = train_test_split(df_normalized, test_size=0.2, random_state=42)

            model.train(train_df)
            predictions = model.predict(test_df)
            accuracy = model.evaluate(test_df, predictions)

            feature_columns = train_df.columns.drop(['Label']).tolist()

            # Prepare dictionary with results for the user-trained model
            user_coefs = {"Bias": model.thetas[0]}
            user_coefs.update({feature_columns[i]: coef for i, coef in enumerate(model.thetas[1:])})

            result_dict = {
                "learning_rate": n_val,
                "accuracy": accuracy,
                "coefficients": user_coefs,
            }
        except Exception as e:
            result_dict = {"error": str(e)}

    # Use the precomputed default results here, do NOT recompute them inside the route
    return render_template("train.html",
                           default_accuracy=default_accuracy,
                           default_coefs=default_coefs,
                           result=result_dict,
                           default_n=default_n)




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
