# Central configuration for the Basketball Analytics project

# ── NBA API Settings ──────────────────────────────────────────
HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Host": "stats.nba.com",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

# ── Seasons ───────────────────────────────────────────────────
CURRENT_SEASON = "2024-25"
SEASONS = ["2022-23", "2023-24", "2024-25"]

# ── Teams ─────────────────────────────────────────────────────
TIMBERWOLVES_ID = 1610612750
LAKERS_ID = 1610612747

# ── Players (Timberwolves focused) ────────────────────────────
PLAYER_IDS = {
    "Anthony Edwards": 1630162,
    "Rudy Gobert":     203497,
    "Julius Randle":   203944,
    "Naz Reid":        1629675,
    "Mike Conley":     201144,
}

# ── Shot Zones ────────────────────────────────────────────────
SHOT_ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
    "Backcourt",
]

# ── Model Settings ────────────────────────────────────────────
RANDOM_STATE    = 42
TEST_SIZE       = 0.2
CV_FOLDS        = 5

# ── Paths ─────────────────────────────────────────────────────
RAW_DATA_PATH       = "data/raw/"
PROCESSED_DATA_PATH = "data/processed/"