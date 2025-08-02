import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def extract_metadata(df: pd.DataFrame):
    return {
        "dtypes": df.dtypes.apply(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "shape": df.shape
    }

def clean_data(df, missing, encode, scale):
    if missing == "drop":
        df = df.dropna()
    else:
        df = df.fillna(df.median(numeric_only=True))

    for col in df.select_dtypes(include=['object']).columns:
        if encode == "label":
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        elif encode == "one-hot":
            df = pd.get_dummies(df, columns=[col])

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler() if scale == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
