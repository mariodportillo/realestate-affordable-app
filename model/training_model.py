import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import json
from tqdm import tqdm
import gzip
import io


class LogisticRegressionRealEstate:
    def __init__(self, learning_rate=0.0001, training_steps=3000):
        self.learning_rate = learning_rate
        self.training_steps = training_steps
        self.thetas = None
        self.stats = {}

    @staticmethod
    def sigmoid(z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def normalize(self, df):
        df = df.copy()
        self.stats = {}
        for col in df.columns:
            if col != 'Label':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / std
                self.stats[col] = (mean, std)
        return df

    def train(self, df):
        df = df.copy()
        df.insert(0, "x0", 1)  # Bias term
        Y = df['Label'].values
        X = df.drop(columns=['Label']).values
        self.thetas = np.zeros(X.shape[1])

        for _ in tqdm(range(self.training_steps), desc="Training steps"):
            predictions = self.sigmoid(np.dot(X, self.thetas))
            errors = Y - predictions
            gradients = np.dot(errors, X)
            self.thetas += self.learning_rate * gradients

    def predict(self, df):
        df = df.copy()
        df.insert(0, "x0", 1)
        X = df.drop(columns=['Label']).values
        predictions = self.sigmoid(np.dot(X, self.thetas))
        return (predictions >= 0.5).astype(int)

    def evaluate(self, df, predictions):
        true_values = df['Label'].values
        return np.mean(predictions == true_values)

    def tune_learning_rates(self, df_train, df_test):
        n_values = [
            1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4,
            1e-3, 5e-3, 1e-2, 5e-2, 1e-1
        ]
        results = {}

        for n in tqdm(n_values, desc="Tuning learning rates"):
            model = LogisticRegressionRealEstate(learning_rate=n, training_steps=1000)
            model.train(df_train)
            predictions = model.predict(df_test)
            accuracy = model.evaluate(df_test, predictions)
            results[n] = accuracy

        return results

    def save_weights(self, filepath="weights.npy"):
        if self.thetas is not None:
            np.save(filepath, self.thetas)

    def save_stats(self, filepath="stats.json"):
        with open(filepath, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def print_coefficients(self, feature_names):
        print(f"Bias: Coefficient = {self.thetas[0]:.4f}")
        for i, coef in enumerate(self.thetas[1:]):
            print(f"Feature {feature_names[i]}: Coefficient = {coef:.4f}")


if __name__ == "__main__":
    with gzip.open("ahs2023n.feather.gz", "rb") as f:
        decompressed_bytes = f.read()
    df = pd.read_feather(io.BytesIO(decompressed_bytes))

    model = LogisticRegressionRealEstate(learning_rate=0.0005)
    df_normalized = model.normalize(df)
    model.save_stats("stats.json")

    train_df, test_df = train_test_split(df_normalized, test_size=0.2, random_state=42)

    model.train(train_df)
    model.save_weights("weights.npy")

    predictions = model.predict(test_df)
    accuracy = model.evaluate(test_df, predictions)
    print(f"Accuracy on test set: {accuracy:.2%}")

    feature_columns = train_df.columns.drop(['Label']).tolist()
    model.print_coefficients(feature_columns)
