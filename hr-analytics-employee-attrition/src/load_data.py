import pandas as pd
from .config import DATA_PATH

def load_raw() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    # Standardize column names (optional)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]
    return df
