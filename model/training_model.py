import pandas as pd
from sklearn.model_selection import train_test_split
import json
import numpy as np
from tqdm import tqdm  # for progress bars
import gzip
import io

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def logistic_regression(df, n=0.0001, training_steps=3_000):
    df = df.copy()
    df.insert(0, "x0", 1)  # Add intercept column with 1s
    thetas = np.zeros(df.shape[1] - 1)
    Y = df['Label'].values
    X = df.drop(columns=['Label']).values

    for _ in tqdm(range(training_steps), desc="Training steps"):
        predictions = sigmoid(np.dot(X, thetas))
        errors = Y - predictions
        gradients = np.dot(errors, X)  # vectorized gradient
        thetas += n * gradients

    return thetas


def test_thetas(df_test, thetas):
    df_test = df_test.copy()
    df_test.insert(0, "x0", 1)
    X = df_test.drop(columns=['Label']).values
    predictions = sigmoid(np.dot(X, thetas))
    return (predictions >= 0.5).astype(int)  # vectorized thresholding


def percent_correct(df_test, predictions):
    true_results = df_test['Label'].values
    return np.mean(predictions == true_results)


def test_different_n(df_train, df_test):
    n_values = [
        1e-6,
        5e-6,
        1e-5,
        5e-5,
        1e-4,
        5e-4,
        1e-3,
        5e-3,
        1e-2,
        5e-2,
        1e-1
    ]
    results = {}

    for n in tqdm(n_values, desc="Learning rates"):
        thetas = logistic_regression(df_train, n, training_steps=1000)
        predictions = test_thetas(df_test, thetas)
        accuracy = percent_correct(df_test, predictions)
        results[n] = accuracy

    return results


def normalize(df):
    df = df.copy()
    stats = {}
    for col in df.columns:
        if col != 'Label':
            mean = df[col].mean()
            std = df[col].std()
            df[col] = (df[col] - mean) / std
            stats[col] = (mean, std)
    return df, stats



if __name__ == "__main__":
    with gzip.open("ahs2023n.feather.gz", "rb") as f:
        decompressed_bytes = f.read()

    df = pd.read_feather(io.BytesIO(decompressed_bytes))
    df_final, stats = normalize(df)

    with open('stats.json', 'w') as f:
        json.dump(stats, f, indent=4)

    train, test = train_test_split(df_final, test_size=0.2, random_state=42)
    thetas = logistic_regression(df_final, 0.0005)
    np.save("weights.npy", thetas)

    feature_columns = train.columns.drop(['Label']).tolist()
    print(f"Bias: Coefficient = {thetas[0]:.4f}")
    for i, coef in enumerate(thetas[1:]):
        print(f"Feature {feature_columns[i]}: Coefficient = {coef:.4f}")
