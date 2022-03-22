# %% tags=["parameters"]
upstream = {
    "make_dataset": {
        "df_wine": "../../data/raw/df_wine.parquet",
    }
}
product = {
    "df_wine_train": "../../data/interim/df_wine_train.parquet",
    "df_wine_test": "../../data/interim/df_wine_test.parquet",
}


# %% tags=[]
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# %% tags=[]
df_wine = pd.read_parquet(Path(upstream["make_dataset"]["df_wine"]))
df_wine

# %% tags=[]
df_wine_train, df_wine_test = train_test_split(df_wine, test_size=0.5, random_state=0)

# %% tags=[]
df_wine_train

# %% tags=[]
df_wine_test

# %% tags=[]
df_wine_train.to_parquet(product["df_wine_train"])
df_wine_test.to_parquet(product["df_wine_test"])