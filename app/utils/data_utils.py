import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

def extract_metadata(df: pd.DataFrame):
    return {
        "dtypes": df.dtypes.apply(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "shape": df.shape
    }


from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import pandas as pd

def clean_data(df, missing, encode, scale):
    # Handle missing values
    if missing == "drop":
        df = df.dropna()
    else:
        df = df.fillna(df.median(numeric_only=True))

    # Encode categorical columns
    object_cols = df.select_dtypes(include=['object', 'bool']).columns
    if encode == "label":
        for col in object_cols:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    elif encode == "one-hot":
        df = pd.get_dummies(df, columns=object_cols, dtype=int)  # Force numeric 0/1

    # Double-check all remaining non-numeric columns and convert if needed
    for col in df.columns:
        if df[col].dtype not in ['int64', 'float64']:
            try:
                df[col] = df[col].astype(float)
            except:
                pass  # Skip columns that can't be safely converted (shouldn't happen here)

    # Normalize numerical features
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    scaler = StandardScaler() if scale == "standard" else MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df
