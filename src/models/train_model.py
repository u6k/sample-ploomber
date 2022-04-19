import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

# %% tags=["parameters"]
upstream = ["build_features"]


# %%
df_wine = pd.read_parquet(upstream["build_features"]["df_wine"])
df_wine


# %%
df_wine_data = df_wine.drop(["target"], axis=1)
df_wine_target = df_wine[["target"]]

df_wine_scaled = pd.DataFrame(
    scale(df_wine_data),
    columns=df_wine_data.columns,
)


# %%
df_wine_scaled


# %%
df_wine_target


# %%
df_wine_train_data, df_wine_test_data, df_wine_train_target, df_wine_test_target = train_test_split(
    df_wine_scaled, df_wine_target,
    test_size=0.5,
    random_state=0
)


# %%
df_wine_train_data


# %%
df_wine_train_target.value_counts()


# %%
df_wine_test_data


# %%
df_wine_test_target.value_counts()


# %%
model = RandomForestClassifier().fit(df_wine_train_data, df_wine_train_target)
model


# %%
df_pred_train = df_wine_train_target.copy()
df_pred_train["pred"] = model.predict(df_wine_train_data)

df_pred_train


# %%
df_pred_test = df_wine_test_target.copy()
df_pred_test["pred"] = model.predict(df_wine_test_data)

df_pred_test


# %%
pickle.dump(model, open(product["model"], "wb"))


# %%
df_wine_train_data.to_parquet(product["df_wine_train_data"])
df_wine_train_target.to_parquet(product["df_wine_train_target"])
df_wine_test_data.to_parquet(product["df_wine_test_data"])
df_wine_test_target.to_parquet(product["df_wine_test_target"])
df_pred_train.to_parquet(product["df_pred_train"])
df_pred_test.to_parquet(product["df_pred_test"])
