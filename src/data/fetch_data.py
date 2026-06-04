# PURPOSE: Pull real NBA data from stats.nba.com.

import os
import time
import pandas as pd

from nba_api.stats.endpoints import (
    ShotChartDetail,
    PlayerGameLog,
    CommonPlayerInfo,
    CommonTeamRoster,
)

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from config import (
    CURRENT_SEASON,
    SEASONS,
    PLAYER_IDS,
    TIMBERWOLVES_ID,
    RAW_DATA_PATH,
)


def _save(df: pd.DataFrame, filename: str) -> None:
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    path = os.path.join(RAW_DATA_PATH, filename)
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} rows → {path}")


def _pause() -> None:
    time.sleep(1)


def fetch_shot_chart(player_id: int, player_name: str, season: str) -> pd.DataFrame:
    print(f"  Fetching shots: {player_name} | {season}")
    try:
        shot_chart = ShotChartDetail(
            team_id=0,
            player_id=player_id,
            season_nullable=season,
            season_type_all_star="Regular Season",
            context_measure_simple="FGA",
        )
        df = shot_chart.get_data_frames()[0]
        df["PLAYER_NAME"] = player_name
        df["SEASON"] = season
        _pause()
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def fetch_all_shot_charts() -> pd.DataFrame:
    print("\nFetching shot charts...")
    all_shots = []
    for player_name, player_id in PLAYER_IDS.items():
        for season in SEASONS:
            df = fetch_shot_chart(player_id, player_name, season)
            if not df.empty:
                all_shots.append(df)
    if all_shots:
        combined = pd.concat(all_shots, ignore_index=True)
        _save(combined, "shot_charts.csv")
        return combined
    return pd.DataFrame()


def fetch_game_log(player_id: int, player_name: str, season: str) -> pd.DataFrame:
    print(f"  Fetching game log: {player_name} | {season}")
    try:
        game_log = PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star="Regular Season",
        )
        df = game_log.get_data_frames()[0]
        df["PLAYER_NAME"] = player_name
        df["SEASON"] = season
        _pause()
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def fetch_all_game_logs() -> pd.DataFrame:
    print("\nFetching game logs...")
    all_logs = []
    for player_name, player_id in PLAYER_IDS.items():
        for season in SEASONS:
            df = fetch_game_log(player_id, player_name, season)
            if not df.empty:
                all_logs.append(df)
    if all_logs:
        combined = pd.concat(all_logs, ignore_index=True)
        _save(combined, "game_logs.csv")
        return combined
    return pd.DataFrame()


def fetch_player_info(player_id: int, player_name: str) -> pd.DataFrame:
    print(f"  Fetching player info: {player_name}")
    try:
        info = CommonPlayerInfo(player_id=player_id)
        df = info.get_data_frames()[0]
        df["PLAYER_NAME"] = player_name
        _pause()
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def fetch_all_player_info() -> pd.DataFrame:
    print("\nFetching player info...")
    all_info = []
    for player_name, player_id in PLAYER_IDS.items():
        df = fetch_player_info(player_id, player_name)
        if not df.empty:
            all_info.append(df)
    if all_info:
        combined = pd.concat(all_info, ignore_index=True)
        _save(combined, "player_info.csv")
        return combined
    return pd.DataFrame()


def fetch_timberwolves_roster() -> pd.DataFrame:
    print("\nFetching Timberwolves roster...")
    try:
        roster = CommonTeamRoster(
            team_id=TIMBERWOLVES_ID,
            season=CURRENT_SEASON,
        )
        df = roster.get_data_frames()[0]
        _save(df, "timberwolves_roster.csv")
        _pause()
        return df
    except Exception as e:
        print(f"  ERROR: {e}")
        return pd.DataFrame()


def fetch_all_data() -> dict:
    print("=" * 60)
    print("BASKETBALL ANALYTICS — DATA COLLECTION")
    print(f"Players: {list(PLAYER_IDS.keys())}")
    print("=" * 60)

    data = {
        "shot_charts": fetch_all_shot_charts(),
        "game_logs":   fetch_all_game_logs(),
        "player_info": fetch_all_player_info(),
        "roster":      fetch_timberwolves_roster(),
    }

    print("\n" + "=" * 60)
    print("COMPLETE")
    for key, df in data.items():
        if not df.empty:
            print(f"  {key}: {len(df)} rows")
    return data


if __name__ == "__main__":
    fetch_all_data()