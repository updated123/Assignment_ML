import shap
import pandas as pd
import joblib
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load .env variables
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",  # Groq-compatible OpenAI endpoint
    api_key=api_key
)

MODEL_DIR = "models"

def explain_batch(df: pd.DataFrame, model_name: str, target_column: str):
    # Load the model and metadata
    model = joblib.load(f"{MODEL_DIR}/{model_name}.pkl")

    # Drop target column if it exists in test data
    feature_df = df.drop(columns=[target_column], errors="ignore")

    # Predict
    predictions = model.predict(feature_df)
    feature_df = feature_df.copy()
    feature_df["prediction"] = predictions

    # SHAP expects only feature inputs (not predictions)
    explainer = shap.Explainer(model.predict, feature_df.drop(columns=["prediction"]))
    shap_values = explainer(feature_df.drop(columns=["prediction"]))

    explanations = []

    for i in range(len(feature_df)):
        row_shap = shap_values[i]
        prediction = feature_df.iloc[i]["prediction"]

        # Top 3 features
        top_indices = abs(row_shap.values).argsort()[::-1][:3]
        top_features = [(feature_df.drop(columns=["prediction"]).columns[j], row_shap.values[j]) for j in top_indices]

        # Prompt
        feature_str = ", ".join(
            f"{name} (SHAP value: {value:.2f})" for name, value in top_features
        )
        prompt = (
            f"Given a model prediction of {'churn' if prediction == 1 else 'no churn'}, "
            f"and the following feature impacts based on SHAP values: {feature_str}, "
            "generate a brief explanation (1-2 sentences) for why the model predicted this outcome."
        )

        # 🔄 Updated OpenAI/Groq-compatible API call
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150,
        )

        explanation = response.choices[0].message.content
        explanations.append(explanation)

    return explanations

