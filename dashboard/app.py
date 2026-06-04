# PURPOSE: Interactive dashboard visualizing shot prediction and player performance models.

import os
import sys
import pandas as pd
import numpy as np
import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_PATH, PLAYER_IDS


# load data

shot_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "shot_features.csv"))
game_df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "game_features.csv"))
game_df["GAME_DATE"] = pd.to_datetime(game_df["GAME_DATE"])

shot_model = joblib.load("data/processed/shot_prediction_model.joblib")

PLAYERS = sorted(shot_df["PLAYER_NAME"].unique().tolist())
SEASONS = sorted(shot_df["SEASON"].unique().tolist())


# court drawing

def draw_court(fig, row=1, col=1):
    court_shapes = [
        dict(type="rect", x0=-250, x1=250, y0=-47.5, y1=422.5,
             line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=-80, x1=80, y0=-47.5, y1=143,
             line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=-60, x1=60, y0=77.5, y1=197.5,
             line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=-7.5, x1=7.5, y0=-7.5, y1=7.5,
             line=dict(color="white", width=2), fillcolor="white"),
        dict(type="line", x0=-30, x1=30, y0=-47.5, y1=-47.5,
             line=dict(color="white", width=2)),
        dict(type="path",
             path="M -220 -47.5 L -220 92.5 Q 0 300 220 92.5 L 220 -47.5",
             line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect", x0=-40, x1=40, y0=-47.5, y1=0,
             line=dict(color="white", width=2), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=-60, x1=60, y0=137.5, y1=257.5,
             line=dict(color="white", width=1, dash="dash"),
             fillcolor="rgba(0,0,0,0)"),
    ]
    for shape in court_shapes:
        shape["xref"] = f"x{col if col > 1 else ''}"
        shape["yref"] = f"y{row if row > 1 else ''}"
        fig.add_shape(**shape)
    return fig


# shot chart

def build_shot_chart(player_name: str, season: str) -> go.Figure:
    df = shot_df[
        (shot_df["PLAYER_NAME"] == player_name) &
        (shot_df["SEASON"] == season)
    ].copy()

    made = df[df["SHOT_MADE_FLAG"] == 1]
    missed = df[df["SHOT_MADE_FLAG"] == 0]

    fg_pct = df["SHOT_MADE_FLAG"].mean()
    three_pct = df[df["IS_THREE"] == 1]["SHOT_MADE_FLAG"].mean()
    clutch_pct = df[df["IS_CLUTCH"] == 1]["SHOT_MADE_FLAG"].mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=made["LOC_X"], y=made["LOC_Y"],
        mode="markers",
        name="Made",
        marker=dict(color="#00C851", size=5, opacity=0.7, symbol="circle"),
        hovertemplate="Made<br>Distance: %{customdata[0]} ft<br>Zone: %{customdata[1]}<extra></extra>",
        customdata=made[["SHOT_DISTANCE", "SHOT_ZONE_BASIC"]].values,
    ))

    fig.add_trace(go.Scatter(
        x=missed["LOC_X"], y=missed["LOC_Y"],
        mode="markers",
        name="Missed",
        marker=dict(color="#FF4444", size=5, opacity=0.5, symbol="x"),
        hovertemplate="Missed<br>Distance: %{customdata[0]} ft<br>Zone: %{customdata[1]}<extra></extra>",
        customdata=missed[["SHOT_DISTANCE", "SHOT_ZONE_BASIC"]].values,
    ))

    draw_court(fig)

    fig.update_layout(
        title=dict(
            text=f"{player_name} — {season} Shot Chart | "
                 f"FG: {fg_pct:.1%} | "
                 f"3P: {three_pct:.1%} | "
                 f"Clutch: {clutch_pct:.1%}",
            font=dict(color="white", size=14),
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(range=[-300, 300], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[-60, 450], showgrid=False, zeroline=False, showticklabels=False,
                   scaleanchor="x", scaleratio=1),
        legend=dict(font=dict(color="white")),
        margin=dict(l=20, r=20, t=60, b=20),
    )

    return fig


# performance trajectory

def build_trajectory(player_name: str) -> go.Figure:
    df = game_df[game_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS"],
        mode="markers",
        name="Actual Points",
        marker=dict(color="rgba(255,255,255,0.3)", size=4),
    ))

    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS_L5"],
        mode="lines",
        name="5-Game Avg",
        line=dict(color="#00C851", width=2),
    ))

    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS_L10"],
        mode="lines",
        name="10-Game Avg",
        line=dict(color="#FFD700", width=2),
    ))

    if "HOT_STREAK" in df.columns:
        hot = df[df["HOT_STREAK"] == 1]
        if len(hot) > 0:
            fig.add_trace(go.Scatter(
                x=hot["GAME_DATE"], y=hot["PTS"],
                mode="markers",
                name="Hot Streak",
                marker=dict(color="#FF6B00", size=8, symbol="star"),
            ))

    fig.update_layout(
        title=dict(
            text=f"{player_name} — Scoring Trajectory",
            font=dict(color="white", size=14),
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)",
                   color="white", title="Points"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


# player comparison

def build_comparison() -> go.Figure:
    summary = game_df.groupby("PLAYER_NAME").agg(
        PTS=("PTS", "mean"),
        REB=("REB", "mean"),
        AST=("AST", "mean"),
        FG_PCT=("FG_PCT", "mean"),
        PLUS_MINUS=("PLUS_MINUS", "mean"),
    ).round(2).reset_index()

    fig = go.Figure()

    metrics = ["PTS", "REB", "AST", "FG_PCT", "PLUS_MINUS"]
    colors = ["#00C851", "#FFD700", "#FF6B00", "#00BFFF", "#FF4444"]

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name=metric,
            x=summary["PLAYER_NAME"],
            y=summary[metric],
            marker_color=color,
        ))

    fig.update_layout(
        title=dict(
            text="Player Comparison — Season Averages",
            font=dict(color="white", size=14),
        ),
        barmode="group",
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=False, color="white"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", color="white"),
        legend=dict(font=dict(color="white")),
        margin=dict(l=40, r=20, t=60, b=40),
    )

    return fig


# feature importance

def build_feature_importance() -> go.Figure:
    features = [
        "SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD",
        "SECONDS_LEFT_IN_PERIOD", "IS_CLUTCH", "IS_THREE",
        "LATE_CLOCK", "IS_CORNER_THREE", "IS_RESTRICTED_AREA",
        "SHOT_DIFFICULTY", "SHOT_ZONE_ENCODED", "COURT_SIDE_ENCODED",
        "DISTANCE_BUCKET_ENCODED", "PLAYER_ENCODED",
    ]

    importance_df = pd.DataFrame({
        "feature": features,
        "importance": shot_model.feature_importances_,
    }).sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker_color="#00C851",
    ))

    fig.update_layout(
        title=dict(
            text="Shot Prediction — Feature Importance",
            font=dict(color="white", size=14),
        ),
        plot_bgcolor="#1a1a2e",
        paper_bgcolor="#1a1a2e",
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.1)", color="white"),
        yaxis=dict(showgrid=False, color="white"),
        margin=dict(l=200, r=20, t=60, b=40),
    )

    return fig


# app layout

app = dash.Dash(__name__)

app.layout = html.Div(
    style={"backgroundColor": "#0f0f23", "minHeight": "100vh", "padding": "20px",
           "fontFamily": "Arial, sans-serif"},
    children=[

        html.Div(
            style={"textAlign": "center", "marginBottom": "30px"},
            children=[
                html.H1("🐺 Timberwolves Basketball Analytics",
                        style={"color": "#00C851", "fontSize": "28px", "margin": "0"}),
                html.P("Shot Prediction & Player Performance Dashboard",
                       style={"color": "rgba(255,255,255,0.6)", "margin": "5px 0 0 0"}),
            ]
        ),

        html.Div(
            style={"display": "flex", "gap": "15px", "marginBottom": "20px",
                   "justifyContent": "center"},
            children=[
                html.Div([
                    html.Label("Player", style={"color": "white", "marginBottom": "5px",
                                                "display": "block"}),
                    dcc.Dropdown(
                        id="player-dropdown",
                        options=[{"label": p, "value": p} for p in PLAYERS],
                        value=PLAYERS[0],
                        style={"width": "220px", "backgroundColor": "#1a1a2e",
                               "color": "black"},
                        clearable=False,
                    ),
                ]),
                html.Div([
                    html.Label("Season", style={"color": "white", "marginBottom": "5px",
                                                "display": "block"}),
                    dcc.Dropdown(
                        id="season-dropdown",
                        options=[{"label": s, "value": s} for s in SEASONS],
                        value=SEASONS[-1],
                        style={"width": "150px", "backgroundColor": "#1a1a2e",
                               "color": "black"},
                        clearable=False,
                    ),
                ]),
            ]
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "15px",
                   "marginBottom": "15px"},
            children=[
                dcc.Graph(id="shot-chart", style={"height": "450px"}),
                dcc.Graph(id="trajectory-chart", style={"height": "450px"}),
            ]
        ),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "15px"},
            children=[
                dcc.Graph(id="comparison-chart",
                          figure=build_comparison(), style={"height": "400px"}),
                dcc.Graph(id="importance-chart",
                          figure=build_feature_importance(), style={"height": "400px"}),
            ]
        ),
    ]
)

#callbacks

@app.callback(
    Output("shot-chart", "figure"),
    Input("player-dropdown", "value"),
    Input("season-dropdown", "value"),
)
def update_shot_chart(player_name, season):
    return build_shot_chart(player_name, season)


@app.callback(
    Output("trajectory-chart", "figure"),
    Input("player-dropdown", "value"),
)
def update_trajectory(player_name):
    return build_trajectory(player_name)

#run

if __name__ == "__main__":
    app.run(debug=True)