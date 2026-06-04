# PURPOSE: Clean and structure raw NBA data for modeling and visualization.

import os
import pandas as pd
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from config import RAW_DATA_PATH, PROCESSED_DATA_PATH


def _load(filename: str) -> pd.DataFrame:
    path = os.path.join(RAW_DATA_PATH, filename)
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows from {filename}")
    return df


def _save(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    path = os.path.join(PROCESSED_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")


def process_shot_charts() -> pd.DataFrame:
    print("\nProcessing shot charts...")
    df = _load("shot_charts.csv")

    # Keep only the columns we need
    cols = [
        "PLAYER_NAME", "SEASON", "GAME_DATE",
        "SHOT_MADE_FLAG",       # target variable: 1 = made, 0 = missed
        "SHOT_TYPE",            # 2PT or 3PT
        "SHOT_ZONE_BASIC",      # Restricted Area, Mid-Range, etc.
        "SHOT_ZONE_AREA",       # Left Side, Right Side, Center
        "SHOT_DISTANCE",        # feet from basket
        "LOC_X", "LOC_Y",       # court coordinates
        "ACTION_TYPE",          # Jump Shot, Layup, Dunk, etc.
        "PERIOD",               # quarter (1-4, 5=OT)
        "MINUTES_REMAINING",    # game situation
        "SECONDS_REMAINING",
    ]
    df = df[cols].copy()

    # Convert game date to datetime
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Add seconds remaining in period (combines minutes + seconds)
    df["SECONDS_LEFT_IN_PERIOD"] = (
        df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]
    )

    # Flag clutch situations: 4th quarter, under 5 minutes
    df["IS_CLUTCH"] = (
        (df["PERIOD"] == 4) & (df["MINUTES_REMAINING"] < 5)
    ).astype(int)

    # Flag 3-point attempts
    df["IS_THREE"] = (df["SHOT_TYPE"] == "3PT Field Goal").astype(int)

    # Drop rows with missing coordinates
    df = df.dropna(subset=["LOC_X", "LOC_Y", "SHOT_DISTANCE"])

    print(f"  Made shots: {df['SHOT_MADE_FLAG'].sum()} "
          f"({df['SHOT_MADE_FLAG'].mean():.1%} FG%)")

    _save(df, "shot_charts_processed.csv")
    return df


def process_game_logs() -> pd.DataFrame:
    print("\nProcessing game logs...")
    df = _load("game_logs.csv")

    # Convert date
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Sort by player and date — critical for time series
    df = df.sort_values(["PLAYER_NAME", "GAME_DATE"]).reset_index(drop=True)

    df["MIN_PLAYED"] = pd.to_numeric(df["MIN"], errors="coerce")

    # Rolling averages — last 5 games and last 10 games
    # These capture recent form rather than season average
    for player in df["PLAYER_NAME"].unique():
        mask = df["PLAYER_NAME"] == player
        for col in ["PTS", "REB", "AST", "FG_PCT", "FG3_PCT", "PLUS_MINUS"]:
            df.loc[mask, f"{col}_L5"] = (
                df.loc[mask, col].rolling(5, min_periods=1).mean()
            )
            df.loc[mask, f"{col}_L10"] = (
                df.loc[mask, col].rolling(10, min_periods=1).mean()
            )

    # Game number within season (for trajectory analysis)
    df["GAME_NUMBER"] = df.groupby(
        ["PLAYER_NAME", "SEASON"]
    ).cumcount() + 1

    # Win flag
    df["WIN"] = (df["WL"] == "W").astype(int)

    _save(df, "game_logs_processed.csv")
    return df


def process_player_info() -> pd.DataFrame:
    print("\nProcessing player info...")
    df = _load("player_info.csv")

    # Keep useful columns
    cols = [
        "PLAYER_NAME", "BIRTHDATE", "HEIGHT", "WEIGHT",
        "POSITION", "DRAFT_YEAR", "DRAFT_NUMBER", "SEASON_EXP",
    ]
    # Only keep columns that exist
    cols = [c for c in cols if c in df.columns]
    df = df[cols].copy()

    # Calculate age from birthdate
    if "BIRTHDATE" in df.columns:
        df["BIRTHDATE"] = pd.to_datetime(df["BIRTHDATE"])
        today = pd.Timestamp.today()
        df["AGE"] = (today - df["BIRTHDATE"]).dt.days // 365

    _save(df, "player_info_processed.csv")
    return df


def process_all() -> dict:
    print("=" * 60)
    print("BASKETBALL ANALYTICS — DATA PROCESSING")
    print("=" * 60)

    data = {
        "shot_charts": process_shot_charts(),
        "game_logs":   process_game_logs(),
        "player_info": process_player_info(),
    }

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    for key, df in data.items():
        print(f"  {key}: {len(df)} rows, {len(df.columns)} columns")

    return data


if __name__ == "__main__":
    process_all()