import pandas as pd
from sklearn.model_selection import train_test_split

# %% tags=["parameters"]
upstream = ["make_dataset"]


# %%
df_wine = pd.read_parquet(upstream["make_dataset"]["df_wine"])
df_wine


# %%
df_wine.to_parquet(product["df_wine"])
