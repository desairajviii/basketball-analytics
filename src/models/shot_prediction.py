# PURPOSE: Train and evaluate a shot prediction model on real NBA tracking data.

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")
from config import PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE, CV_FOLDS


FEATURES = [
    "SHOT_DISTANCE",
    "LOC_X",
    "LOC_Y",
    "PERIOD",
    "SECONDS_LEFT_IN_PERIOD",
    "IS_CLUTCH",
    "IS_THREE",
    "LATE_CLOCK",
    "IS_CORNER_THREE",
    "IS_RESTRICTED_AREA",
    "SHOT_DIFFICULTY",
    "SHOT_ZONE_ENCODED",
    "COURT_SIDE_ENCODED",
    "DISTANCE_BUCKET_ENCODED",
    "PLAYER_ENCODED",
]

TARGET = "SHOT_MADE_FLAG"
MODEL_PATH = "data/processed/shot_prediction_model.joblib"


def load_features() -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(PROCESSED_DATA_PATH, "shot_features.csv")
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} shots")

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    return X, y


def train_model(X: pd.DataFrame, y: pd.Series) -> XGBClassifier:
    print("\nTraining XGBoost classifier...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model, X_train, X_test, y_train, y_test


def evaluate_model(
    model: XGBClassifier,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict:
    print("\nEvaluating model...")

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=CV_FOLDS,
        scoring="accuracy",
    )

    print(f"\n  Accuracy:          {accuracy:.1%}")
    print(f"  F1 Score:          {f1:.3f}")
    print(f"  CV Accuracy:       {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Missed", "Made"]))

    print("  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"    True Missed:  {cm[0][0]:>4}  False Made: {cm[0][1]:>4}")
    print(f"    False Missed: {cm[1][0]:>4}  True Made:  {cm[1][1]:>4}")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
    }


def feature_importance(model: XGBClassifier) -> pd.DataFrame:
    print("\n  Feature Importance:")

    importance_df = pd.DataFrame({
        "feature": FEATURES,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    for _, row in importance_df.iterrows():
        bar = "█" * int(row["importance"] * 100)
        print(f"    {row['feature']:<30} {row['importance']:.3f} {bar}")

    return importance_df


def save_model(model: XGBClassifier) -> None:
    os.makedirs("data/processed", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\n  Model saved → {MODEL_PATH}")


def run() -> dict:
    print("=" * 60)
    print("BASKETBALL ANALYTICS — SHOT PREDICTION MODEL")
    print("=" * 60)

    X, y = load_features()
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test)
    feature_importance(model)
    save_model(model)

    print("\n" + "=" * 60)
    print("SHOT PREDICTION COMPLETE")
    print(f"  Final Accuracy: {metrics['accuracy']:.1%}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    run()