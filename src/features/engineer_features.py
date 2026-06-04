# PURPOSE: Create model-ready features from processed NBA data.

import os
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from config import PROCESSED_DATA_PATH


def _load(filename: str) -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {filename}")
    return df


def _save(df: pd.DataFrame, filename: str) -> None:
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")


def engineer_shot_features() -> pd.DataFrame:
    print("\nEngineering shot features...")
    df = _load("shot_charts_processed.csv")

    df["DISTANCE_BUCKET"] = pd.cut(
        df["SHOT_DISTANCE"],
        bins=[0, 4, 10, 16, 23, 100],
        labels=["Paint", "Short Mid", "Mid Range", "Long 2", "Three+"],
        right=True,
    )

    def classify_court_side(loc_x):
        if loc_x < -50:
            return "Left"
        elif loc_x > 50:
            return "Right"
        else:
            return "Center"

    df["COURT_SIDE"] = df["LOC_X"].apply(classify_court_side)

    df["IS_CORNER_THREE"] = (
        df["SHOT_ZONE_BASIC"].isin(["Left Corner 3", "Right Corner 3"])
    ).astype(int)

    df["IS_RESTRICTED_AREA"] = (
        df["SHOT_ZONE_BASIC"] == "Restricted Area"
    ).astype(int)

    df["LATE_CLOCK"] = (df["SECONDS_LEFT_IN_PERIOD"] < 4).astype(int)

    df["SHOT_DIFFICULTY"] = (
        (df["SHOT_DISTANCE"] - 10).clip(lower=0) * 0.05
        + df["IS_CLUTCH"] * 0.3
        + df["LATE_CLOCK"] * 0.2
        - df["IS_CORNER_THREE"] * 0.1
        - df["IS_RESTRICTED_AREA"] * 0.5
    ).clip(lower=0)

    df["SHOT_ZONE_ENCODED"]       = pd.Categorical(df["SHOT_ZONE_BASIC"]).codes
    df["COURT_SIDE_ENCODED"]      = pd.Categorical(df["COURT_SIDE"]).codes
    df["DISTANCE_BUCKET_ENCODED"] = pd.Categorical(df["DISTANCE_BUCKET"]).codes
    df["PLAYER_ENCODED"]          = pd.Categorical(df["PLAYER_NAME"]).codes

    zone_summary = df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"].agg(
        ["count", "mean"]
    ).round(3)
    zone_summary.columns = ["Attempts", "FG%"]
    print(zone_summary.to_string())

    _save(df, "shot_features.csv")
    return df


def engineer_game_features() -> pd.DataFrame:
    print("\nEngineering game features...")
    df = _load("game_logs_processed.csv")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    for player in df["PLAYER_NAME"].unique():
        mask = df["PLAYER_NAME"] == player

        if "PTS_L5" in df.columns and "PTS_L10" in df.columns:
            df.loc[mask, "PERFORMANCE_TREND"] = (
                df.loc[mask, "PTS_L5"] - df.loc[mask, "PTS_L10"]
            )

        df.loc[mask, "CONSISTENCY"] = (
            df.loc[mask, "PTS"].rolling(10, min_periods=3).std()
        )

        if "MIN_PLAYED" in df.columns:
            df.loc[mask, "FATIGUE_PROXY"] = (
                df.loc[mask, "MIN_PLAYED"].rolling(5, min_periods=1).sum()
            )

        pts_l20 = df.loc[mask, "PTS"].rolling(20, min_periods=5).mean()
        pts_l5  = df.loc[mask, "PTS"].rolling(5, min_periods=1).mean()

        df.loc[mask, "HOT_STREAK"]  = (pts_l5 > pts_l20 * 1.15).astype(int)
        df.loc[mask, "COLD_STREAK"] = (pts_l5 < pts_l20 * 0.85).astype(int)

        df.loc[mask, "NEXT_GAME_PTS"] = df.loc[mask, "PTS"].shift(-1)

    df["SEASON_PROGRESS"] = df["GAME_NUMBER"] / df.groupby(
        ["PLAYER_NAME", "SEASON"]
    )["GAME_NUMBER"].transform("max")

    for player in df["PLAYER_NAME"].unique():
        mask = df["PLAYER_NAME"] == player
        df.loc[mask, "WIN_RATE_L10"] = (
            df.loc[mask, "WIN"].rolling(10, min_periods=1).mean()
        )

    df = df.dropna(subset=["NEXT_GAME_PTS"])

    trend_summary = df.groupby("PLAYER_NAME")["PERFORMANCE_TREND"].mean().round(2)
    print(trend_summary.to_string())

    _save(df, "game_features.csv")
    return df


def engineer_all_features() -> dict:
    print("=" * 60)
    print("BASKETBALL ANALYTICS — FEATURE ENGINEERING")
    print("=" * 60)

    data = {
        "shot_features": engineer_shot_features(),
        "game_features": engineer_game_features(),
    }

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows, {len(df.columns)} columns")

    return data


if __name__ == "__main__":
    engineer_all_features()