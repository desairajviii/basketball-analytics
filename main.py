# PURPOSE: Single entry point — runs the full basketball analytics pipeline.

import sys
import time

from src.data.fetch_data import fetch_all_data
from src.data.process_data import process_all
from src.features.engineer_features import engineer_all_features
from src.models.shot_prediction import run as run_shot_model
from src.models.player_performance import run as run_performance_model


def main():
    start = time.time()

    print("=" * 60)
    print("TIMBERWOLVES BASKETBALL ANALYTICS")
    print("Full Pipeline Run")
    print("=" * 60)

    print("\n[1/5] Fetching data from NBA Stats API...")
    fetch_all_data()

    print("\n[2/5] Processing raw data...")
    process_all()

    print("\n[3/5] Engineering features...")
    engineer_all_features()

    print("\n[4/5] Training shot prediction model...")
    run_shot_model()

    print("\n[5/5] Training player performance models...")
    run_performance_model()

    elapsed = time.time() - start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print(f"Total time: {minutes}m {seconds}s")
    print("=" * 60)
    print("\nTo launch the dashboard run:")
    print("  python3 dashboard/app.py")
    print("Then open: http://127.0.0.1:8050")


if __name__ == "__main__":
    main()