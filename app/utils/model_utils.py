import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import os

MODELS = {
    "logistic": LogisticRegression,
    "randomforest": RandomForestClassifier,
    "xgboost": XGBClassifier
}

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train(df, model_list, target):
    print(model_list)
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {}
    for name in model_list:
        model = MODELS[name]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")

        report = classification_report(y_test, y_pred, output_dict=True)
        conf = confusion_matrix(y_test, y_pred).tolist()

        results[name] = {
            "metrics": report,
            "confusion_matrix": conf
        }
    return results

# model_utils.py
def predict(df, model_name):
    model = joblib.load(f"{MODEL_DIR}/{model_name}.pkl")

    # Get feature names used during training
    trained_features = model.feature_names_in_

    # Drop target column and its one-hot versions if they exist
    df = df[[col for col in df.columns if col in trained_features]]

    # Predict
    df["prediction"] = model.predict(df)

    return df

