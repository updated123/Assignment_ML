import os
import openai
import pandas as pd

openai.api_key = os.getenv("GROQ_API_KEY")

def explain_batch(df: pd.DataFrame, target: str):
    prompts = []
    for _, row in df.iterrows():
        row_data = row.to_dict()
        prompt = (
            f"Explain why this customer {'is' if row_data['prediction']==1 else 'is not'} "
            f"likely to churn given the data: {row_data}"
        )
        prompts.append(prompt)

    explanations = []
    for prompt in prompts:
        try:
            response = openai.ChatCompletion.create(
                model="llama3-70b-8192",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant explaining customer churn."},
                    {"role": "user", "content": prompt}
                ]
            )
            explanations.append(response["choices"][0]["message"]["content"])
        except Exception:
            explanations.append("LLM explanation unavailable.")

    return explanations
