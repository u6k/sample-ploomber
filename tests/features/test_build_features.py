import pandas as pd


def test_product():
    df_wine = pd.read_parquet("/var/output/data/interim/df_wine.parquet")

    assert len(df_wine) > 0
    for t in df_wine["target"]:
        assert t in [0, 1, 2]
