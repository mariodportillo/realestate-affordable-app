import pandas as pd
import numpy as np


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

# def evaluate_affordability(zipcode_input, all_listings_df, user_inputs, thetas, norm_stats):
#     """
#     Given a zipcode and user financial data, evaluate each listing and return the probability of affordability.
#     """
#
#     # Clean zip_code column: convert to int, drop invalid rows
#     all_listings_df = all_listings_df.copy()
#     all_listings_df['zip_code'] = pd.to_numeric(all_listings_df['zip_code'], errors='coerce')
#     all_listings_df = all_listings_df.dropna(subset=['zip_code'])
#     all_listings_df['zip_code'] = all_listings_df['zip_code'].astype(int)
#
#     # Filter listings by ZIP code
#     listings_in_zip = all_listings_df[all_listings_df['zip_code'] == zipcode_input].copy()
#     if listings_in_zip.empty:
#         print("No listings found in that ZIP code.")
#         return []
#
#     predictions = []
#
#     for idx, row in listings_in_zip.iterrows():
#         try:
#             # Validate and convert list_price
#             list_price = float(row['list_price']) if pd.notna(row['list_price']) else 0
#             if list_price <= 0:
#                 continue  # Skip invalid prices
#
#             # Extract user inputs
#             down_payment = float(user_inputs['down_payment'])
#             interest = float(user_inputs['annual_interest_rate'])
#             loan_term_months = int(user_inputs['loan_term_in_years'] * 12)
#             monthly_debts = float(user_inputs['monthly_debts'])
#             hincome = float(user_inputs['grossIncome'])
#
#             # Calculate total housing cost (custom function you have)
#             tothcamt = monthly_expenses(row, interest, loan_term_months, monthly_debts, down_payment)
#
#             # Bathrooms: fallback to 0 if missing or invalid
#             bathrooms = 0
#             if pd.notna(row.get('full_baths')):
#                 try:
#                     bathrooms = int(float(row['full_baths']))
#                 except (ValueError, TypeError):
#                     bathrooms = 0
#
#             # Bedrooms: fallback to 0 if missing or invalid
#             bedrooms = 0
#             if pd.notna(row.get('beds')):
#                 try:
#                     bedrooms = int(float(row['beds']))
#                     bedrooms = max(0, min(6, bedrooms))  # clamp between 0 and 6
#                 except (ValueError, TypeError):
#                     bedrooms = 0
#
#             # Unitsize is the square footage (assumed numeric), handle missing
#             unitsize = float(row['sqft']) if pd.notna(row.get('sqft')) else 0
#
#             # Normalize or scale inputs here if your model expects normalized inputs
#             # Example (if your model expects normalized):
#             # Otherwise just raw or scaled as per your training
#
#             # Construct feature vector with intercept term 1 first (bias)
#             monthly_income = hincome / 12 if hincome > 0 else 1
#             affordable_ratio = float(tothcamt) / monthly_income
#
#             features = {
#                 'HINCP': hincome,
#                 'BEDROOMS': bedrooms,
#                 'BATHROOMS': bathrooms,
#                 'TOTHCAMT': tothcamt,
#                 'UNITSIZE': unitsize,
#                 'AFFORDABLE': affordable_ratio
#             }
#
#             normalized_features = []
#             for name, value in features.items():
#                 mean, std = norm_stats[name]
#                 norm_val = (value - mean) / std if std > 0 else 0
#                 normalized_features.append(norm_val)
#
#             x_vec = np.array([1] + normalized_features)
#
#             # Calculate probability using sigmoid and your learned thetas
#             prob = sigmoid(np.dot(thetas, x_vec))
#
#             predictions.append({
#                 "listing_id": row.get("id", idx),
#                 "list_price": list_price,
#                 "property_url": row.get("property_url", None),
#                 "tothcamt": tothcamt,
#                 "affordable_ratio": affordable_ratio,
#                 "hincome": hincome,
#                 "bathrooms": bathrooms,
#                 "bedrooms": bedrooms,
#                 "sqft": unitsize,
#                 "predicted_probability": prob,
#                 "affordable": int(prob >= 0.5)
#             })
#
#         except Exception as e:
#             continue  # skip problematic rows
#
#     # Sort predictions by descending affordability probability
#     predictions = sorted(predictions, key=lambda x: x['predicted_probability'], reverse=True)
#     return predictions

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
    df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce')
    df = df[df['zip_code'] == int(zipcode_input)]
    df = df[pd.to_numeric(df['list_price'], errors='coerce') > 0]

    if df.empty:
        return []

    # Convert relevant fields to numeric
    df['list_price'] = pd.to_numeric(df['list_price'], errors='coerce').fillna(0)
    df['beds'] = pd.to_numeric(df.get('beds', 0), errors='coerce').fillna(0).astype(int).clip(0, 6)
    df['full_baths'] = pd.to_numeric(df.get('full_baths', 0), errors='coerce').fillna(0).astype(int)
    df['sqft'] = pd.to_numeric(df.get('sqft', 0), errors='coerce').fillna(0)
    df['hoa_fee'] = pd.to_numeric(df.get('hoa_fee', 0), errors='coerce').fillna(0)
    df['tax'] = pd.to_numeric(df.get('tax', np.nan), errors='coerce')

    # Estimate missing taxes
    estimated_tax = df['list_price'] * (0.0125 + 0.0025)
    df['tax'] = df['tax'].fillna(estimated_tax)
    df['tax_monthly'] = df['tax'] / 12.0

    # Fixed insurance
    df['insurance'] = 200.0

    # Mortgage calculation (vectorized)
    loan_amount = (df['list_price'] - down_payment).clip(lower=0)
    monthly_rate = interest / 1200

    with np.errstate(divide='ignore', invalid='ignore'):
        df['mortgage'] = (
            loan_amount * (monthly_rate * (1 + monthly_rate) ** loan_term_months)
            / ((1 + monthly_rate) ** loan_term_months - 1)
        )
        df['mortgage'] = df['mortgage'].replace([np.inf, -np.inf], 0).fillna(0)

    # Total monthly housing cost
    df['TOTHCAMT'] = df['tax_monthly'] + df['insurance'] + df['hoa_fee'] + df['mortgage'] + monthly_debts

    # Affordability ratio
    df['affordable_ratio'] = df['TOTHCAMT'] / monthly_income

    # Add features
    df['HINCP'] = hincome
    df['BEDROOMS'] = df['beds']
    df['BATHROOMS'] = df['full_baths']
    df['UNITSIZE'] = df['sqft']
    df['AFFORDABLE'] = df['affordable_ratio']

    # Feature normalization
    feature_cols = ['HINCP', 'BEDROOMS', 'BATHROOMS', 'TOTHCAMT', 'UNITSIZE', 'AFFORDABLE']
    normalized = []

    for col in feature_cols:
        value = df[col]
        mean, std = norm_stats.get(col, (0, 1))
        mean = float(mean)
        std = float(std) if std > 0 else 1
        norm_val = (value - mean) / std
        normalized.append(norm_val)

    # Feature matrix
    X = np.stack(normalized, axis=1)  # shape (n, 6)
    X = np.hstack([np.ones((X.shape[0], 1)), X])  # add intercept

    # Predictions
    z = X.dot(thetas)
    probs = sigmoid(z)
    df['predicted_probability'] = probs
    df['affordable'] = (probs >= 0.5).astype(int)

    # Build response records
    df['listing_id'] = df.get('id', df.index)
    df['property_url'] = df.get('property_url', None)

    results = df[
        ['listing_id', 'list_price', 'property_url', 'TOTHCAMT', 'affordable_ratio',
         'HINCP', 'BATHROOMS', 'BEDROOMS', 'UNITSIZE', 'predicted_probability', 'affordable']
    ].sort_values(by='predicted_probability', ascending=False)

    return results.to_dict(orient='records')

