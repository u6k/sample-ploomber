import pandas as pd


def test_df_pred():
    df_pred = pd.read_parquet("/var/output/data/processed/df_pred.parquet")

    assert len(df_pred) == 1
    assert len(df_pred.columns) == 1
    assert df_pred["pred"] is not None
