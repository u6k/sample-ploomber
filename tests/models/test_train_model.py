import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def test_product():
    # Load file
    df_wine_train_data = pd.read_parquet("/var/output/data/processed/df_wine_train_data.parquet")
    df_wine_train_target = pd.read_parquet("/var/output/data/processed/df_wine_train_target.parquet")
    df_wine_test_data = pd.read_parquet("/var/output/data/processed/df_wine_test_data.parquet")
    df_wine_test_target = pd.read_parquet("/var/output/data/processed/df_wine_test_target.parquet")
    df_pred_train = pd.read_parquet("/var/output/data/processed/df_pred_train.parquet")
    df_pred_test = pd.read_parquet("/var/output/data/processed/df_pred_test.parquet")

    with open("/var/output/models/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("/var/output/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    # Assert
    assert len(df_wine_train_data) > 0
    assert len(df_wine_train_data.columns) == 13

    assert len(df_wine_train_target) == len(df_wine_train_data)
    assert len(df_wine_train_target.columns) == 1
    for t in df_wine_train_target["target"]:
        assert t in [0, 1, 2]

    assert len(df_wine_test_data) > 0
    assert len(df_wine_test_data.columns) == 13

    assert len(df_wine_test_target) == len(df_wine_test_data)
    assert len(df_wine_test_target.columns) == 1
    for t in df_wine_test_target["target"]:
        assert t in [0, 1, 2]

    assert len(df_pred_train) == len(df_wine_train_target)
    assert len(df_pred_train.columns) == 2
    for t in df_pred_train["target"]:
        assert t in [0, 1, 2]
    for p in df_pred_train["pred"]:
        assert p in [0, 1, 2]

    assert len(df_pred_test) == len(df_wine_test_data)
    assert len(df_pred_test.columns) == 2
    for t in df_pred_test["target"]:
        assert t in [0, 1, 2]
    for p in df_pred_test["pred"]:
        assert p in [0, 1, 2]

    assert type(model) is RandomForestClassifier
    assert type(scaler) is StandardScaler
