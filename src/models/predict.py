import pickle

import pandas as pd

# %% tags=["parameters"]
upstream = ["build_features"]
product = None
model_path = None
scaler_path = None
predict_target_index = None


# %% データを読み込む
df_wine = pd.read_parquet(upstream["build_features"]["df_wine"]) \
    .loc[predict_target_index:predict_target_index]
df_wine


# %% モデルを読み込む
with open(model_path, "rb") as f:
    model = pickle.load(f)

model


# %%
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

scaler


# %% 推論のためデータ変換する
df_wine_data = df_wine.drop(["target"], axis=1)
df_wine_data


# %%
df_wine_scaled = pd.DataFrame(
    scaler.transform(df_wine_data),
    columns=df_wine_data.columns
)

df_wine_scaled


# %% 推論する
df_pred = df_wine_data.copy()
df_pred["pred"] = model.predict(df_wine_scaled)

df_pred = df_pred.drop(df_wine_data.columns, axis=1)

df_pred


# %% 推論結果を出力する
df_pred.to_parquet(product["df_pred"])
