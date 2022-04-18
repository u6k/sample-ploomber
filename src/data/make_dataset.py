from pathlib import Path

import pandas as pd
from sklearn.datasets import load_wine

# %% tags=["parameters"]
upstream = None
product = {
    "df_wine": "../../data/raw/df_wine.parquet",
}


# %%
data_wine = load_wine()
data_wine


# %%
df_wine = pd.DataFrame(
    data_wine.data,
    columns=data_wine.feature_names
)
df_wine["target"] = data_wine.target

df_wine


# %%
path_df_wine = Path(product["df_wine"]).resolve()
path_df_wine


# %%
df_wine.to_parquet(path_df_wine)
