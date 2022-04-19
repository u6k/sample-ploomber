import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


def test_df_wine_train_data():
    df_wine_train_data = pd.read_parquet("/var/output/data/processed/df_wine_train_data.parquet")

    assert len(df_wine_train_data) > 0
    assert len(df_wine_train_data.columns) == 13


def test_df_wine_train_target():
    df_wine_train_target = pd.read_parquet("/var/output/data/processed/df_wine_train_target.parquet")

    assert len(df_wine_train_target) > 0
    assert len(df_wine_train_target.columns) == 1
    for t in df_wine_train_target["target"]:
        assert t in [0, 1, 2]


def test_df_wine_test_data():
    df_wine_test_data = pd.read_parquet("/var/output/data/processed/df_wine_test_data.parquet")

    assert len(df_wine_test_data) > 0
    assert len(df_wine_test_data.columns) == 13


def test_df_wine_test_target():
    df_wine_test_target = pd.read_parquet("/var/output/data/processed/df_wine_test_target.parquet")

    assert len(df_wine_test_target) > 0
    assert len(df_wine_test_target.columns) == 1
    for t in df_wine_test_target["target"]:
        assert t in [0, 1, 2]


def test_df_pred_train():
    df_pred_train = pd.read_parquet("/var/output/data/processed/df_pred_train.parquet")

    assert len(df_pred_train) > 0
    assert len(df_pred_train.columns) == 2
    for t in df_pred_train["target"]:
        assert t in [0, 1, 2]
    for p in df_pred_train["pred"]:
        assert p in [0, 1, 2]


def test_df_pred_test():
    df_pred_test = pd.read_parquet("/var/output/data/processed/df_pred_test.parquet")

    assert len(df_pred_test) > 0
    assert len(df_pred_test.columns) == 2
    for t in df_pred_test["target"]:
        assert t in [0, 1, 2]
    for p in df_pred_test["pred"]:
        assert p in [0, 1, 2]


def test_model():
    with open("/var/output/models/model.pkl", "rb") as f:
        model = pickle.load(f)

    assert type(model) is RandomForestClassifier


def test_scaler():
    with open("/var/output/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    assert type(scaler) is StandardScaler
