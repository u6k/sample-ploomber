import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# %% tags=["parameters"]
upstream = {
    "build_features": {
        "df_wine_train": "../../data/interim/df_wine_train.parquet",
        "df_wine_test": "../../data/interim/df_wine_test.parquet",
    }
}
product = {
    "model": "../../models/model.pkl",
    "pred_train": "../../data/processed/df_pred_train.parquet",
    "pred_test": "../../data/processed/df_pred_test.parquet",
}


# %%
df_wine_train = pd.read_parquet(upstream["build_features"]["df_wine_train"])
df_wine_test = pd.read_parquet(upstream["build_features"]["df_wine_test"])


# %%
df_wine_train


# %%
df_wine_test


# %%
model = RandomForestClassifier().fit(
    df_wine_train.drop(["target"], axis=1),
    df_wine_train["target"]
)

model


# %%
pred_wine_train = model.predict(df_wine_train.drop(["target"], axis=1))
pred_wine_test = model.predict(df_wine_test.drop(["target"], axis=1))


# %%
df_pred_train = df_wine_train[["target"]]
df_pred_train["pred"] = pred_wine_train

df_pred_test = df_wine_test[["target"]]
df_pred_test["pred"] = pred_wine_test


# %%
df_pred_train


# %%
df_pred_test


# %%
pickle.dump(model, open(product["model"], "wb"))


# %%
df_pred_train.to_parquet(product["pred_train"])
df_pred_test.to_parquet(product["pred_test"])
