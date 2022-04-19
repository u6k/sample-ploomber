import pandas as pd
from sklearn.datasets import load_wine

# %% tags=["parameters"]
upstream = None


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
df_wine.to_parquet(product["df_wine"])
