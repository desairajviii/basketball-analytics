# PURPOSE: Train per-player performance prediction models using game log features.

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from config import PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE


FEATURES = [
    "GAME_NUMBER",
    "SEASON_PROGRESS",
    "MIN_PLAYED",
    "PTS_L5",
    "PTS_L10",
    "REB_L5",
    "AST_L5",
    "FG_PCT_L5",
    "FG3_PCT_L5",
    "PLUS_MINUS_L5",
    "PERFORMANCE_TREND",
    "CONSISTENCY",
    "FATIGUE_PROXY",
    "HOT_STREAK",
    "COLD_STREAK",
    "WIN_RATE_L10",
]

TARGET = "NEXT_GAME_PTS"
MODELS_DIR = "data/processed/player_models/"


def load_features() -> pd.DataFrame:
    path = os.path.join(PROCESSED_DATA_PATH, "game_features.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} game records")
    return df


def train_player_model(
    player_name: str,
    df_player: pd.DataFrame,
) -> dict:
    df_clean = df_player[FEATURES + [TARGET]].dropna()

    if len(df_clean) < 30:
        print(f"  {player_name}: insufficient data ({len(df_clean)} games), skipping")
        return None

    X = df_clean[FEATURES]
    y = df_clean[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
    )

    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model":       model,
        "mae":         mae,
        "r2":          r2,
        "n_games":     len(df_clean),
        "avg_pts":     y.mean(),
        "y_test":      y_test,
        "y_pred":      y_pred,
    }


def print_player_summary(player_name: str, result: dict) -> None:
    print(f"\n  {player_name}")
    print(f"    Games:        {result['n_games']}")
    print(f"    Avg Points:   {result['avg_pts']:.1f}")
    print(f"    MAE:          {result['mae']:.1f} pts")
    print(f"    R² Score:     {result['r2']:.3f}")

    errors = np.abs(result["y_test"].values - result["y_pred"])
    within_5 = (errors <= 5).mean()
    within_10 = (errors <= 10).mean()
    print(f"    Within 5 pts: {within_5:.1%}")
    print(f"    Within 10pts: {within_10:.1%}")


def feature_importance(player_name: str, model: XGBRegressor) -> None:
    importance_df = pd.DataFrame({
        "feature":    FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(5)

    print(f"    Top features:")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"      {row['feature']:<25} {row['importance']:.3f} {bar}")


def save_models(results: dict) -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    for player_name, result in results.items():
        if result is None:
            continue
        safe_name = player_name.replace(" ", "_").lower()
        path = os.path.join(MODELS_DIR, f"{safe_name}_model.joblib")
        joblib.dump(result["model"], path)
    print(f"\n  Models saved → {MODELS_DIR}")


def run() -> dict:
    print("=" * 60)
    print("BASKETBALL ANALYTICS — PLAYER PERFORMANCE MODEL")
    print("=" * 60)

    df = load_features()
    results = {}

    for player_name in df["PLAYER_NAME"].unique():
        df_player = df[df["PLAYER_NAME"] == player_name].copy()
        result = train_player_model(player_name, df_player)

        if result:
            results[player_name] = result
            print_player_summary(player_name, result)
            feature_importance(player_name, result["model"])

    save_models(results)

    print("\n" + "=" * 60)
    print("PERFORMANCE PREDICTION COMPLETE")
    print(f"  Players modeled: {len(results)}")

    maes = [r["mae"] for r in results.values()]
    print(f"  Average MAE:     {np.mean(maes):.1f} pts")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run()