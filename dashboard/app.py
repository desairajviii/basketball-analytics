# PURPOSE: Interactive Timberwolves analytics dashboard — shot prediction and player performance.

import os
import sys
import pandas as pd
import numpy as np
import joblib

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROCESSED_DATA_PATH, PLAYER_IDS

# Load Data

shot_df  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "shot_features.csv"))
game_df  = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, "game_features.csv"))
game_df["GAME_DATE"] = pd.to_datetime(game_df["GAME_DATE"])

shot_model   = joblib.load("data/processed/shot_prediction_model.joblib")
MODELS_DIR   = "data/processed/player_models/"

PLAYERS  = sorted(shot_df["PLAYER_NAME"].unique().tolist())
SEASONS  = sorted(shot_df["SEASON"].unique().tolist())

# Timberwolves Brand Colors 

NAVY    = "#1D428A"
GREEN   = "#236192"
SILVER  = "#C0C0C0"
WHITE   = "#FFFFFF"
BG_DARK = "#0B1120"
BG_CARD = "#111827"
BG_PANEL = "#1A2540"
ACCENT  = "#4A90D9"
MADE    = "#22C55E"
MISSED  = "#EF4444"
HOT     = "#F97316"

PLOTLY_LAYOUT = dict(
    plot_bgcolor  = BG_CARD,
    paper_bgcolor = BG_CARD,
    font          = dict(color=SILVER, family="Georgia, serif", size=12),
    legend        = dict(
        font=dict(color=SILVER, size=11),
        bgcolor="rgba(0,0,0,0)",
    ),
)

# Court Drawing 

def get_court_shapes():
    return [
        dict(type="rect",   x0=-250, x1=250,   y0=-47.5, y1=422.5,
             line=dict(color=SILVER, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect",   x0=-80,  x1=80,    y0=-47.5, y1=143,
             line=dict(color=SILVER, width=1.5), fillcolor="rgba(29,66,138,0.15)"),
        dict(type="circle", x0=-60,  x1=60,    y0=77.5,  y1=197.5,
             line=dict(color=SILVER, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=-7.5, x1=7.5,   y0=-7.5,  y1=7.5,
             line=dict(color=SILVER, width=2),   fillcolor=SILVER),
        dict(type="line",   x0=-30,  x1=30,    y0=-47.5, y1=-47.5,
             line=dict(color=SILVER, width=2)),
        dict(type="path",
             path="M -220 -47.5 L -220 92.5 Q 0 300 220 92.5 L 220 -47.5",
             line=dict(color=SILVER, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="rect",   x0=-40,  x1=40,    y0=-47.5, y1=0,
             line=dict(color=SILVER, width=1.5), fillcolor="rgba(0,0,0,0)"),
        dict(type="circle", x0=-60,  x1=60,    y0=137.5, y1=257.5,
             line=dict(color=SILVER, width=1, dash="dash"),
             fillcolor="rgba(0,0,0,0)"),
        dict(type="line",   x0=-220, x1=-220,  y0=-47.5, y1=92.5,
             line=dict(color=SILVER, width=1.5)),
        dict(type="line",   x0=220,  x1=220,   y0=-47.5, y1=92.5,
             line=dict(color=SILVER, width=1.5)),
    ]


# Shot Chart

def build_shot_chart(player_name: str, season: str) -> go.Figure:
    df = shot_df[
        (shot_df["PLAYER_NAME"] == player_name) &
        (shot_df["SEASON"] == season)
    ].copy()

    made   = df[df["SHOT_MADE_FLAG"] == 1]
    missed = df[df["SHOT_MADE_FLAG"] == 0]

    fg_pct     = df["SHOT_MADE_FLAG"].mean()
    three_df   = df[df["IS_THREE"] == 1]
    three_pct  = three_df["SHOT_MADE_FLAG"].mean() if len(three_df) > 0 else 0
    clutch_df  = df[df["IS_CLUTCH"] == 1]
    clutch_pct = clutch_df["SHOT_MADE_FLAG"].mean() if len(clutch_df) > 0 else 0
    total      = len(df)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=missed["LOC_X"], y=missed["LOC_Y"],
        mode="markers",
        name=f"Missed ({len(missed)})",
        marker=dict(color=MISSED, size=6, opacity=0.55, symbol="x",
                    line=dict(width=1, color=MISSED)),
        hovertemplate="<b>Missed</b><br>Distance: %{customdata[0]} ft<br>Zone: %{customdata[1]}<extra></extra>",
        customdata=missed[["SHOT_DISTANCE", "SHOT_ZONE_BASIC"]].values,
    ))

    fig.add_trace(go.Scatter(
        x=made["LOC_X"], y=made["LOC_Y"],
        mode="markers",
        name=f"Made ({len(made)})",
        marker=dict(color=MADE, size=7, opacity=0.75, symbol="circle",
                    line=dict(width=0.5, color="rgba(0,0,0,0.3)")),
        hovertemplate="<b>Made</b><br>Distance: %{customdata[0]} ft<br>Zone: %{customdata[1]}<extra></extra>",
        customdata=made[["SHOT_DISTANCE", "SHOT_ZONE_BASIC"]].values,
    ))

    for shape in get_court_shapes():
        fig.add_shape(**shape)

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{player_name}  ·  {season}",
            font=dict(color=WHITE, size=15, family="Georgia, serif"),
            x=0.02,
        ),
        xaxis=dict(range=[-260, 260], showgrid=False, zeroline=False,
                   showticklabels=False, fixedrange=True),
        yaxis=dict(range=[-60, 440], showgrid=False, zeroline=False,
                   showticklabels=False, scaleanchor="x", scaleratio=1,
                   fixedrange=True),
        annotations=[
            dict(x=-240, y=410, text=f"FG%<br><b>{fg_pct:.1%}</b>",
                 showarrow=False, font=dict(color=MADE, size=13), align="center"),
            dict(x=-240, y=360, text=f"3P%<br><b>{three_pct:.1%}</b>",
                 showarrow=False, font=dict(color=ACCENT, size=13), align="center"),
            dict(x=-240, y=310, text=f"Clutch%<br><b>{clutch_pct:.1%}</b>",
                 showarrow=False, font=dict(color=HOT, size=13), align="center"),
            dict(x=-240, y=260, text=f"Attempts<br><b>{total}</b>",
                 showarrow=False, font=dict(color=SILVER, size=13), align="center"),
        ],
        margin=dict(l=20, r=20, t=50, b=20),
    )

    return fig


# Performance Trajectory 

def build_trajectory(player_name: str) -> go.Figure:
    df = game_df[game_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS"],
        mode="markers",
        name="Game Score",
        marker=dict(color="rgba(192,192,192,0.25)", size=5),
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Points: %{y}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS_L5"],
        mode="lines",
        name="5-Game Avg",
        line=dict(color=MADE, width=2.5),
        hovertemplate="5-Game Avg: %{y:.1f}<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=df["GAME_DATE"], y=df["PTS_L10"],
        mode="lines",
        name="10-Game Avg",
        line=dict(color="#FFD700", width=2, dash="dot"),
        hovertemplate="10-Game Avg: %{y:.1f}<extra></extra>",
    ))

    if "HOT_STREAK" in df.columns:
        hot = df[df["HOT_STREAK"] == 1]
        if len(hot) > 0:
            fig.add_trace(go.Scatter(
                x=hot["GAME_DATE"], y=hot["PTS"],
                mode="markers",
                name="Hot Streak",
                marker=dict(color=HOT, size=9, symbol="star",
                            line=dict(width=1, color=WHITE)),
                hovertemplate="<b>Hot Streak</b><br>Points: %{y}<extra></extra>",
            ))

    avg_pts = df["PTS"].mean()
    fig.add_hline(
        y=avg_pts,
        line=dict(color="rgba(192,192,192,0.3)", width=1, dash="dash"),
        annotation_text=f"Avg {avg_pts:.1f}",
        annotation_font=dict(color=SILVER, size=11),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{player_name}  ·  Scoring Trajectory",
            font=dict(color=WHITE, size=15, family="Georgia, serif"),
            x=0.02,
        ),
        xaxis=dict(showgrid=False, color=SILVER, tickformat="%b %Y"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   color=SILVER, title="Points", zeroline=False),
    )

    return fig


# Next Game Prediction 

def build_prediction_panel(player_name: str) -> go.Figure:
    safe_name  = player_name.replace(" ", "_").lower()
    model_path = os.path.join(MODELS_DIR, f"{safe_name}_model.joblib")

    df_player = game_df[game_df["PLAYER_NAME"] == player_name].sort_values("GAME_DATE")

    FEATURES = [
        "GAME_NUMBER", "SEASON_PROGRESS", "MIN_PLAYED",
        "PTS_L5", "PTS_L10", "REB_L5", "AST_L5",
        "FG_PCT_L5", "FG3_PCT_L5", "PLUS_MINUS_L5",
        "PERFORMANCE_TREND", "CONSISTENCY", "FATIGUE_PROXY",
        "HOT_STREAK", "COLD_STREAK", "WIN_RATE_L10",
    ]

    fig = go.Figure()

    if not os.path.exists(model_path):
        fig.add_annotation(text="Model not available", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=SILVER, size=14))
        fig.update_layout(**PLOTLY_LAYOUT)
        return fig

    model       = joblib.load(model_path)
    last_game   = df_player[FEATURES].dropna().iloc[-1:]

    if last_game.empty:
        fig.add_annotation(text="Insufficient data", x=0.5, y=0.5,
                           showarrow=False, font=dict(color=SILVER, size=14))
        fig.update_layout(**PLOTLY_LAYOUT)
        return fig

    prediction  = model.predict(last_game)[0]
    season_avg  = df_player["PTS"].mean()
    recent_avg  = df_player["PTS_L5"].iloc[-1] if "PTS_L5" in df_player.columns else season_avg
    consistency = df_player["CONSISTENCY"].iloc[-1] if "CONSISTENCY" in df_player.columns else 5
    if pd.isna(consistency):
        consistency = 5

    low  = max(0, prediction - consistency)
    high = prediction + consistency

    fig.add_trace(go.Bar(
        x=["Season Avg", "Recent Avg\n(L5)", "Predicted\nNext Game"],
        y=[season_avg, recent_avg, prediction],
        marker_color=[SILVER, ACCENT, MADE],
        text=[f"{season_avg:.1f}", f"{recent_avg:.1f}", f"{prediction:.1f}"],
        textposition="outside",
        textfont=dict(color=WHITE, size=13),
        width=0.5,
    ))

    fig.add_shape(
        type="rect",
        x0=1.7, x1=2.3, y0=low, y1=high,
        fillcolor="rgba(34,197,94,0.15)",
        line=dict(color=MADE, width=1, dash="dot"),
    )

    fig.add_annotation(
        x=2, y=high + 1,
        text=f"Range: {low:.0f}–{high:.0f} pts",
        showarrow=False,
        font=dict(color=MADE, size=11),
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text=f"{player_name}  ·  Next Game Prediction",
            font=dict(color=WHITE, size=15, family="Georgia, serif"),
            x=0.02,
        ),
        xaxis=dict(showgrid=False, color=SILVER),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                   color=SILVER, title="Points", zeroline=False,
                   range=[0, high + 8]),
        showlegend=False,
    )

    return fig


# Player Comparison 

def build_comparison() -> go.Figure:
    summary = game_df.groupby("PLAYER_NAME").agg(
        PTS        = ("PTS",        "mean"),
        REB        = ("REB",        "mean"),
        AST        = ("AST",        "mean"),
        FG_PCT     = ("FG_PCT",     "mean"),
        PLUS_MINUS = ("PLUS_MINUS", "mean"),
    ).round(2).reset_index()

    metrics = ["PTS", "REB", "AST", "FG_PCT", "PLUS_MINUS"]
    colors  = [MADE, "#FFD700", HOT, ACCENT, SILVER]

    fig = go.Figure()

    for metric, color in zip(metrics, colors):
        fig.add_trace(go.Bar(
            name       = metric,
            x          = summary["PLAYER_NAME"],
            y          = summary[metric],
            marker_color = color,
            opacity    = 0.85,
        ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Player Comparison  ·  Season Averages",
            font=dict(color=WHITE, size=15, family="Georgia, serif"),
            x=0.02,
        ),
        barmode  = "group",
        xaxis    = dict(showgrid=False, color=SILVER, tickangle=-20),
        yaxis    = dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                        color=SILVER, zeroline=False),
    )

    return fig


# Feature Importance 

def build_feature_importance() -> go.Figure:
    features = [
        "SHOT_DISTANCE", "LOC_X", "LOC_Y", "PERIOD",
        "SECONDS_LEFT_IN_PERIOD", "IS_CLUTCH", "IS_THREE",
        "LATE_CLOCK", "IS_CORNER_THREE", "IS_RESTRICTED_AREA",
        "SHOT_DIFFICULTY", "SHOT_ZONE_ENCODED", "COURT_SIDE_ENCODED",
        "DISTANCE_BUCKET_ENCODED", "PLAYER_ENCODED",
    ]

    importance_df = pd.DataFrame({
        "feature":    features,
        "importance": shot_model.feature_importances_,
    }).sort_values("importance", ascending=True)

    colors = [
        MADE if imp > 0.1 else ACCENT if imp > 0.03 else SILVER
        for imp in importance_df["importance"]
    ]

    fig = go.Figure(go.Bar(
        x             = importance_df["importance"],
        y             = importance_df["feature"],
        orientation   = "h",
        marker_color  = colors,
        text          = [f"{v:.3f}" for v in importance_df["importance"]],
        textposition  = "outside",
        textfont      = dict(color=SILVER, size=10),
    ))

    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(
            text="Shot Prediction  ·  Feature Importance",
            font=dict(color=WHITE, size=15, family="Georgia, serif"),
            x=0.02,
        ),
        xaxis = dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)",
                     color=SILVER, range=[0, 0.65]),
        yaxis = dict(showgrid=False, color=SILVER),
        margin = dict(l=220, r=60, t=50, b=40),
    )

    return fig


# Stat Cards 

def get_stat_cards(player_name: str, season: str) -> list:
    df      = shot_df[(shot_df["PLAYER_NAME"] == player_name) &
                      (shot_df["SEASON"] == season)]
    gdf = game_df[
    (game_df["PLAYER_NAME"] == player_name) &
    (game_df["SEASON"] == season)
]

    fg_pct      = df["SHOT_MADE_FLAG"].mean() if len(df) > 0 else 0
    three_df    = df[df["IS_THREE"] == 1]
    three_pct   = three_df["SHOT_MADE_FLAG"].mean() if len(three_df) > 0 else 0
    clutch_df   = df[df["IS_CLUTCH"] == 1]
    clutch_pct  = clutch_df["SHOT_MADE_FLAG"].mean() if len(clutch_df) > 0 else 0
    avg_pts     = gdf["PTS"].mean() if len(gdf) > 0 else 0
    avg_reb     = gdf["REB"].mean() if len(gdf) > 0 else 0
    avg_ast     = gdf["AST"].mean() if len(gdf) > 0 else 0

    cards = [
        ("PPG",     f"{avg_pts:.1f}",    MADE),
        ("RPG",     f"{avg_reb:.1f}",    ACCENT),
        ("APG",     f"{avg_ast:.1f}",    HOT),
        ("FG%",     f"{fg_pct:.1%}",     "#FFD700"),
        ("3P%",     f"{three_pct:.1%}",  SILVER),
        ("Clutch%", f"{clutch_pct:.1%}", "#EE82EE"),
    ]

    return [
        html.Div(
            style={
                "backgroundColor": BG_PANEL,
                "borderRadius":    "10px",
                "padding":         "14px 18px",
                "textAlign":       "center",
                "borderTop":       f"3px solid {color}",
                "minWidth":        "90px",
            },
            children=[
                html.P(label, style={"color": SILVER, "fontSize": "11px",
                                     "margin": "0", "letterSpacing": "1px",
                                     "textTransform": "uppercase"}),
                html.P(value, style={"color": color, "fontSize": "22px",
                                     "fontWeight": "bold", "margin": "4px 0 0 0",
                                     "fontFamily": "Georgia, serif"}),
            ]
        )
        for label, value, color in cards
    ]


# App Layout 

app = dash.Dash(__name__, title="Wolves Analytics")

app.layout = html.Div(
    style={
        "backgroundColor": BG_DARK,
        "minHeight":        "100vh",
        "padding":          "24px 32px",
        "fontFamily":       "Georgia, serif",
    },
    children=[

        # Header
        html.Div(
            style={"marginBottom": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "center",
                           "gap": "12px", "marginBottom": "4px"},
                    children=[
                        html.Div("🐺", style={"fontSize": "32px"}),
                        html.H1(
                            "TIMBERWOLVES ANALYTICS",
                            style={"color": WHITE, "fontSize": "24px",
                                   "margin": "0", "letterSpacing": "3px",
                                   "fontFamily": "Georgia, serif"},
                        ),
                    ]
                ),
                html.P(
                    "Shot Prediction · Player Performance · Trajectory Analysis",
                    style={"color": "rgba(192,192,192,0.5)", "margin": "0",
                           "fontSize": "12px", "letterSpacing": "2px",
                           "paddingLeft": "44px"},
                ),
                html.Hr(style={"borderColor": "rgba(192,192,192,0.1)",
                               "marginTop": "16px", "marginBottom": "0"}),
            ]
        ),

        # Top Bordered section
        html.Div(
            style={
                "border":        f"1px solid rgba(192,192,192,0.15)",
                "borderRadius":  "12px",
                "padding":       "20px",
                "marginBottom":  "16px",
                "backgroundColor": "rgba(255,255,255,0.02)",
            },
            children=[

                # Player dropdown + stat cards
                html.Div(
                    style={"display": "flex", "gap": "20px",
                           "alignItems": "flex-end", "marginBottom": "20px",
                           "flexWrap": "wrap"},
                    children=[
                        html.Div([
                            html.Label(
                                "PLAYER",
                                style={"color": SILVER, "fontSize": "10px",
                                       "letterSpacing": "2px", "display": "block",
                                       "marginBottom": "6px"},
                            ),
                            dcc.Dropdown(
                                id="player-dropdown",
                                options=[{"label": p, "value": p} for p in PLAYERS],
                                value="Anthony Edwards",
                                style={"width": "230px",
                                       "backgroundColor": BG_PANEL,
                                       "color": "black",
                                       "border": f"1px solid {NAVY}"},
                                clearable=False,
                            ),
                        ]),
                        html.Div(
                            id="stat-cards",
                            style={"display": "flex", "gap": "10px",
                                   "flexWrap": "wrap"},
                        ),
                    ]
                ),

                # 3 charts row
                html.Div(
                    style={"display": "grid",
                           "gridTemplateColumns": "1.2fr 0.8fr 0.8fr",
                           "gap": "16px"},
                    children=[

                        # Shot chart with season dropdown inside
                        html.Div(
                            style={
                                "border":          f"1px solid rgba(192,192,192,0.1)",
                                "borderRadius":    "8px",
                                "padding":         "12px",
                                "backgroundColor": BG_CARD,
                            },
                            children=[
                                html.Div(
                                    style={"display": "flex",
                                           "alignItems": "center",
                                           "gap": "12px",
                                           "marginBottom": "8px"},
                                    children=[
                                        html.Label(
                                            "SEASON",
                                            style={"color": SILVER,
                                                   "fontSize": "10px",
                                                   "letterSpacing": "2px"},
                                        ),
                                        dcc.Dropdown(
                                            id="season-dropdown",
                                            options=[{"label": s, "value": s}
                                                     for s in SEASONS],
                                            value="2024-25",
                                            style={"width": "130px",
                                                   "backgroundColor": BG_PANEL,
                                                   "color": "black",
                                                   "border": f"1px solid {NAVY}"},
                                            clearable=False,
                                        ),
                                    ]
                                ),
                                dcc.Graph(
                                    id="shot-chart",
                                    style={"height": "440px"},
                                    config={"displayModeBar": False},
                                ),
                            ]
                        ),

                        dcc.Graph(
                            id="trajectory-chart",
                            style={"height": "500px"},
                            config={"displayModeBar": False},
                        ),

                        dcc.Graph(
                            id="prediction-panel",
                            style={"height": "500px"},
                            config={"displayModeBar": False},
                        ),
                    ]
                ),
            ]
        ),

        # Constant 2 charts
        html.Div(
            style={"display": "grid",
                   "gridTemplateColumns": "1fr 1fr",
                   "gap": "16px"},
            children=[
                html.Div(
                    style={
                        "border":          f"1px solid rgba(192,192,192,0.15)",
                        "borderRadius":    "12px",
                        "padding":         "4px",
                        "backgroundColor": "rgba(255,255,255,0.02)",
                    },
                    children=[
                        dcc.Graph(
                            id="comparison-chart",
                            figure=build_comparison(),
                            style={"height": "380px"},
                            config={"displayModeBar": False},
                        ),
                    ]
                ),
                html.Div(
                    style={
                        "border":          f"1px solid rgba(192,192,192,0.15)",
                        "borderRadius":    "12px",
                        "padding":         "4px",
                        "backgroundColor": "rgba(255,255,255,0.02)",
                    },
                    children=[
                        dcc.Graph(
                            id="importance-chart",
                            figure=build_feature_importance(),
                            style={"height": "380px"},
                            config={"displayModeBar": False},
                        ),
                    ]
                ),
            ]
        ),

        # Footer
        html.Div(
            style={"textAlign": "center", "marginTop": "24px",
                   "color": "rgba(192,192,192,0.3)", "fontSize": "11px",
                   "letterSpacing": "1px"},
            children="Data sourced from NBA Stats API · "
                     "Models trained on 3 seasons of tracking data",
        ),
    ]
)


# Callbacks

@app.callback(
    Output("shot-chart",       "figure"),
    Output("trajectory-chart", "figure"),
    Output("prediction-panel", "figure"),
    Output("stat-cards",       "children"),
    Input("player-dropdown",   "value"),
    Input("season-dropdown",   "value"),
)
def update_all(player_name, season):
    return (
        build_shot_chart(player_name, season),
        build_trajectory(player_name),
        build_prediction_panel(player_name),
        get_stat_cards(player_name, season),
    )


# Run

if __name__ == "__main__":
    app.run(debug=True)