import pandas as pd

# %% tags=["parameters"]
upstream = ["make_dataset"]
product = None


# %%
df_wine = pd.read_parquet(upstream["make_dataset"]["df_wine"])
df_wine


# %%
df_wine.to_parquet(product["df_wine"])
