from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from app.utils import data_utils, model_utils, llm_utils
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
def root():
    with open("static/index.html") as f:
        return f.read()

@app.post("/upload")
def upload_dataset(file: UploadFile = File(...), target_column: str = Form(...)):
    df = pd.read_csv(file.file)
    df.to_csv(f"{UPLOAD_DIR}/raw.csv", index=False)
    metadata = data_utils.extract_metadata(df)
    return {
        "metadata": metadata,
        "columns": df.columns.tolist(),
        "preview": df.head().to_dict()
    }

@app.post("/data_cleaning")
def clean_dataset(
    missing: str = Form("drop"),
    encode: str = Form("one-hot"),
    scale: str = Form("standard")
):
    df = pd.read_csv(f"{UPLOAD_DIR}/raw.csv")
    cleaned_df = data_utils.clean_data(df, missing, encode, scale)
    cleaned_df.to_csv(f"{UPLOAD_DIR}/cleaned.csv", index=False)
    return {
        "preview": cleaned_df.head().to_dict(),
        "columns": cleaned_df.columns.tolist()
    }

@app.post("/train_model")
async def train_models(
    models: str = Form("logistic,randomforest,xgboost"),
    target_column: str = Form(...)
):
    # ✅ CORRECTED: Removed premature return
    print("Models received:", models)
    print("Target column received:", target_column)

    # ✅ Load cleaned data
    df = pd.read_csv(f"{UPLOAD_DIR}/cleaned.csv")

    # ✅ Train selected models
    results = model_utils.train(df, models.split(","), target_column)
    
    # ✅ Return results as JSON
    return results
@app.post("/predict")
def predict(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    target_column: str = Form(...)
):
    test_df = pd.read_csv(file.file)
    test_df.to_csv(f"{UPLOAD_DIR}/test.csv", index=False)

    # Drop target column if present (should not be in test features)
    if target_column in test_df.columns:
        test_df = test_df.drop(columns=[target_column])

    preds = model_utils.predict(test_df, model_name)

    if isinstance(preds, pd.DataFrame):
        preds = preds.iloc[:, 0]

    pred_df = test_df.copy()
    pred_df["prediction"] = preds

    # ✅ Pass target_column to exclude it in explanations
    explanations = llm_utils.explain_batch(test_df, model_name, target_column)

    pred_df["Explanation"] = explanations

    return pred_df.to_dict(orient="records")
