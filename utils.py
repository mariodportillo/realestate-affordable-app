import pandas as pd
import numpy as np
import json

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def calculate_monthly_mortgage(initial_amount, annual_interest_rate, number_of_months):
    """
    Calculate the monthly mortgage payment based on the loan amount,
    annual interest rate, and number of months.
    """
    monthly_interest_rate = annual_interest_rate / (12 * 100)
    monthly_payment = (
            initial_amount
            * (monthly_interest_rate * (1 + monthly_interest_rate) ** number_of_months)
            / ((1 + monthly_interest_rate) ** number_of_months - 1)
    )
    return monthly_payment


def monthly_expenses(row, annual_interest_rate, loan_term_in_years, monthly_debts, down_payment) -> float:
    """
    Estimate the monthly expenses for a property, including mortgage,
    property tax, insurance, and HOA fees.
    """
    tax = float(row['tax'])
    listing_price = float(row['list_price'])
    hoa = float(row['hoa_fee'])
    insurance = 200

    if not isinstance(tax, (int, float)):
        tax = listing_price * 0.0125 + listing_price * 0.0025
    tax = tax / 12  # 1% annual property tax

    number_of_months = loan_term_in_years * 12
    if listing_price <= down_payment:
        mortgage = 0
    else:
        mortgage = calculate_monthly_mortgage(listing_price - down_payment, annual_interest_rate, number_of_months)

    return round(tax + insurance + hoa + mortgage + monthly_debts, 2)


def ask_for_inputs(zipcodes: list) -> dict:
    def get_float(prompt, max_decimals=2, min_value=None):
        while True:
            user_input = input(prompt)
            try:
                value = float(user_input)

                # Check decimal places
                if '.' in user_input:
                    decimals = user_input.split('.')[1]
                    if len(decimals) > max_decimals:
                        print(f"Please enter a number with at most {max_decimals} decimal places.")
                        continue

                # Check minimum value
                if min_value is not None and value < min_value:
                    print(f"Please enter a value greater than or equal to {min_value}.")
                    continue

                return round(value, max_decimals)
            except ValueError:
                print("Invalid input. Please enter a numeric value.")

    def get_int(prompt, min_value=None):
        while True:
            user_input = input(prompt)
            try:
                value = int(user_input)
                if min_value is not None and value < min_value:
                    print(f"Please enter a value greater than or equal to {min_value}.")
                    continue
                return value
            except ValueError:
                print("Invalid input. Please enter a whole number.")

    def get_zipcode(prompt, min_value=None):
        val = get_int(prompt, min_value)
        while True:
            if len(str(val)) > 5 or val not in zipcodes:
                print("Please put in a valid 5 digit zipcode.")
                val = get_int(prompt, min_value)
            else:
                break
        return val

    results = {
        'grossIncome': get_float("What is your gross income? ", max_decimals=2, min_value=0),
        'monthly_debts': get_float("What are your monthly debts? ", max_decimals=2, min_value=0),
        'down_payment': get_float("How much could you put down for a downpayment(in dollars)? ", max_decimals=2,
                                  min_value=0),
        'zipcode': get_zipcode("Which zipcode are you interested in? ", min_value=1),
        'annual_interest_rate': get_float("What are the current interest rates in %? ", max_decimals=2, min_value=0),
        'loan_term_in_years': get_int("What is your expected loan term in years? ", min_value=1),

    }

    return results


def find_zip_codes(user_inputs: dict, json_path="data/zip_mean_prices.json", metadata_path="data/zip_metadata.json") -> dict:
    gross_income = user_inputs["grossIncome"] - (user_inputs["monthly_debts"] * 12)
    down = user_inputs["down_payment"]
    monthly_rate = (user_inputs["annual_interest_rate"] / 100) / 12
    term = user_inputs["loan_term_in_years"]
    months = term * 12

    if gross_income <= 0:
        return {}

    # Load mean home prices per ZIP code
    with open(json_path, "r") as f:
        zip_price_means = json.load(f)

    # Load city/state info per ZIP code
    with open(metadata_path, "r") as f:
        zip_metadata = json.load(f)

    results = []

    for zip_code, mean_price in zip_price_means.items():
        principal = max(mean_price - down, 0)
        print(f"ZIP: {zip_code}, mean_price: {mean_price}, down: {down}, principal: {principal}")

        if principal == 0:
            total_home_cost_year = 0
        else:
            try:
                M = (principal * monthly_rate * (1 + monthly_rate) ** months) / ((1 + monthly_rate) ** months - 1)
                total_home_cost_year = M * 12
            except ZeroDivisionError:
                total_home_cost_year = float("inf")

        affordability = total_home_cost_year / gross_income
        affordability = 1 / (1 + affordability)

        if zip_code in zip_metadata:
            city = zip_metadata[zip_code].get("city", "Unknown")
            state = zip_metadata[zip_code].get("state", "Unknown")

            results.append({
                "zip": zip_code,
                "city": city,
                "state": state,
                "score": affordability
            })

    # Sort by affordability (lower score = more affordable)
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

    # Return top 200, converted into a dict
    top_200 = sorted_results[:200]
    return {entry["zip"]: {
        "score": entry["score"],
        "city": entry["city"],
        "state": entry["state"]
    } for entry in top_200}


def evaluate_affordability(zipcode_input, all_listings_df, user_inputs, thetas, norm_stats):
    """
    Vectorized, type-safe affordability evaluator for property listings.
    Ensures all numerical operations are done between valid float/int types.
    """

    # Safely extract and convert user inputs
    try:
        hincome = float(user_inputs.get('grossIncome', 0))
        monthly_income = max(hincome / 12, 1)
        down_payment = float(user_inputs.get('down_payment', 0))
        interest = float(user_inputs.get('annual_interest_rate', 0))
        loan_term_years = int(user_inputs.get('loan_term_in_years', 30))
        loan_term_months = loan_term_years * 12
        monthly_debts = float(user_inputs.get('monthly_debts', 0))
    except (TypeError, ValueError):
        return []

    # Filter and clean listings by zip code
    df = all_listings_df.copy()
    df['ZIP_CODE'] = pd.to_numeric(df['ZIP_CODE'], errors='coerce')
    df = df[df['ZIP_CODE'] == int(zipcode_input)]
    df = df[pd.to_numeric(df['LIST_PRICE'], errors='coerce') > 0]

    if df.empty:
        return []

    # Convert relevant fields to numeric
    df['LIST_PRICE'] = pd.to_numeric(df['LIST_PRICE'], errors='coerce').fillna(0)
    df['BEDS'] = pd.to_numeric(df.get('BEDS', 0), errors='coerce').fillna(0).astype(int).clip(0, 6)
    df['FULL_BATHS'] = pd.to_numeric(df.get('FULL_BATHS', 0), errors='coerce').fillna(0).astype(int)
    df['SQFT'] = pd.to_numeric(df.get('SQFT', 0), errors='coerce').fillna(0)
    df['HOA_FEE'] = pd.to_numeric(df.get('HOA_FEE', 0), errors='coerce').fillna(0)
    df['TAX'] = pd.to_numeric(df.get('TAX', np.nan), errors='coerce')

    # Estimate missing taxes
    estimated_tax = df['LIST_PRICE'] * (0.0125 + 0.0025)
    df['TAX'] = df['TAX'].fillna(estimated_tax)
    df['TAX_MONTHLY'] = df['TAX'] / 12.0

    # Fixed insurance
    df['INSURANCE'] = 200.0

    # Mortgage calculation (vectorized)
    loan_amount = (df['LIST_PRICE'] - down_payment).clip(lower=0)
    monthly_rate = interest / 1200

    with np.errstate(divide='ignore', invalid='ignore'):
        df['MORTGAGE'] = (
            loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term_months)
            / ((1 + monthly_rate) ** loan_term_months - 1)
        )
        df['MORTGAGE'] = df['MORTGAGE'].replace([np.inf, -np.inf], 0).fillna(0)

    # Total monthly housing cost
    df['TOTHCAMT'] = df['TAX_MONTHLY'] + df['INSURANCE'] + df['HOA_FEE'] + df['MORTGAGE'] + monthly_debts

    # Affordability ratio
    df['AFFORDABLE_RATIO'] = df['TOTHCAMT'] / monthly_income

    # Add features
    df['HINCP'] = hincome
    df['BEDROOMS'] = df['BEDS']
    df['BATHROOMS'] = df['FULL_BATHS']
    df['UNITSIZE'] = df['SQFT']
    df['AFFORDABLE'] = df['AFFORDABLE_RATIO']

    # Feature normalization
    feature_cols = ['HINCP', 'BEDROOMS', 'BATHROOMS', 'TOTHCAMT', 'UNITSIZE', 'AFFORDABLE']
    normalized = []

    for col in feature_cols:
        value = df[col]
        mean, std = norm_stats.get(col, (0.0, 1.0))
        try:
            mean = float(mean)
        except (TypeError, ValueError):
            mean = 0.0
        try:
            std = float(std)
            if std <= 0:
                std = 1.0
        except (TypeError, ValueError):
            std = 1.0
        norm_val = (value - mean) / std
        normalized.append(norm_val)

    # Feature matrix
    X = np.stack(normalized, axis=1)  # shape (n, 6)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept

    # Predictions
    z = X.dot(thetas)
    probs = sigmoid(z)
    df['PREDICTED_PROBABILITY'] = probs
    df['AFFORDABLE'] = (probs >= 0.5).astype(int)

    # Build response records
    df['LISTING_ID'] = df.get('ID', df.index)
    df['PROPERTY_URL'] = df.get('PROPERTY_URL', None)

    mask = down_payment > df['LIST_PRICE']
    df.loc[mask, 'PREDICTED_PROBABILITY'] = 1.0
    df.loc[mask, 'AFFORDABLE'] = 1

    results = df[
        ['LISTING_ID', 'LIST_PRICE', 'PROPERTY_URL', 'TOTHCAMT', 'AFFORDABLE_RATIO',
         'HINCP', 'BATHROOMS', 'BEDROOMS', 'UNITSIZE', 'PREDICTED_PROBABILITY', 'AFFORDABLE']
    ].sort_values(by='PREDICTED_PROBABILITY', ascending=False)

    return results.to_dict(orient='records')


