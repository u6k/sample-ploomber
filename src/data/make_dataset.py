# %% tags=["parameters"]
upstream = None
product = {
    "df_wine": "../../data/raw/df_wine.parquet",
}

# %% tags=[]
from pathlib import Path

from sklearn.datasets import load_wine
import pandas as pd

# %% tags=[]
data_wine = load_wine()
data_wine

# %% tags=[]
df_wine = pd.DataFrame(
    data_wine.data,
    columns=data_wine.feature_names
)
df_wine["target"] = data_wine.target

df_wine

# %% tags=[]
path_df_wine = Path(product["df_wine"]).resolve()
path_df_wine

# %% tags=[]
df_wine.to_parquet(path_df_wine)
