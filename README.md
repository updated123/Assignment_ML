Step 1:
pip install -r requirements.txt

Step 2:
uvicorn app.main:app --reload


Prompt Explaination:

**SHAP + LLM Explanation
This tool explains why a machine learning model made a prediction.
For each row:
It uses SHAP to find the top 3 features affecting the prediction.
Then, it sends a prompt to a Groq LLM to generate a short, human-friendly explanation.


Prompt Example:

Given a model prediction of [churn/no churn], and the following feature impacts based on SHAP values:
Feature1 (SHAP value: x.xx), Feature2 (SHAP value: y.yy), Feature3 (SHAP value: z.zz),
generate a brief explanation (1-2 sentences) for why the model predicted this outcome.


