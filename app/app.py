from dash import Dash, html, dcc, Input, Output, State
import pandas as pd
import plotly.express as px
import numpy as np

# ====================== LOAD DATA ======================
df = pd.read_csv("dash_dataset.csv")

pred_features = [
    "Income", "Recency", "MntWines", "MntMeatProducts", "MntFishProducts",
    "NumWebPurchases", "NumStorePurchases", "NumCatalogPurchases"
]

# ====================== RAW → Z-SCORE PIPELINE ======================
# These parameters replicate the exact StandardScaler fitted in the cleaning notebooks.
# Pipeline: winsorize(5/95%) → log1p (Mnt* columns only) → StandardScaler

LOG1P_COLS = {"MntWines", "MntMeatProducts", "MntFishProducts"}

scaler_mu = {
    "Income":               51748.393837,
    "Recency":                 49.094643,
    "MntWines":                 4.695636,
    "MntMeatProducts":          4.134491,
    "MntFishProducts":          2.525189,
    "NumWebPurchases":          4.018750,
    "NumStorePurchases":        5.769643,
    "NumCatalogPurchases":      2.592411,
}
scaler_sigma = {
    "Income":               19915.809518,
    "Recency":                 28.542433,
    "MntWines":                 1.735344,
    "MntMeatProducts":          1.513024,
    "MntFishProducts":          1.640990,
    "NumWebPurchases":          2.470470,
    "NumStorePurchases":        3.148209,
    "NumCatalogPurchases":      2.653070,
}

INCOME_MU    = scaler_mu["Income"]
INCOME_SIGMA = scaler_sigma["Income"]

def z_to_raw_income(z):
    return z * INCOME_SIGMA + INCOME_MU

def raw_to_zscore(col, raw_value):
    """Convert a raw user input to the same z-score space as dash_dataset."""
    val = float(raw_value)
    if col in LOG1P_COLS:
        val = np.log1p(max(val, 0))
    return (val - scaler_mu[col]) / scaler_sigma[col]

# ====================== CLUSTER → SEGMENT MAPPING ======================
cluster_to_segment = {}
for model in ["kmeans_cluster", "SOM_Cluster", "DBSCAN_Cluster"]:
    mapping = df.groupby(model)["Segment"].agg(lambda x: x.mode()[0] if not x.empty else "Unknown")
    cluster_to_segment[model] = mapping.to_dict()

def make_label(model, c):
    if model == "DBSCAN_Cluster" and c == -1:
        return "Cluster -1: Outliers / Noise"
    return f"Cluster {c}: {cluster_to_segment[model].get(c, 'Unknown')}"

# ====================== MARKETING RECOMMENDATIONS ======================
marketing_recs = {
    "High-Value Buyers": "Target with premium wine offers, exclusive catalogs, and VIP loyalty events.",
    "Digital Shoppers":  "Focus on web promotions, fast delivery, and online-exclusive deals.",
    "Budget Conscious":  "Emphasize deals, discounts, value bundles, and price-match guarantees.",
    "Passive Segment":   "Re-engage with win-back campaigns, personalized reminders, and special reactivation offers."
}

# ====================== SEGMENT DESCRIPTION CARDS ======================
segment_descriptions = {
    "High-Value Buyers": ("💎", "#7c3aed", "High income, high spending on wines and meat products. Frequent catalog and store shoppers. Best targets for premium offers and loyalty programs."),
    "Digital Shoppers":  ("🛒", "#0ea5e9", "Moderate income, prefer buying online. High web purchase frequency. Respond well to digital promotions, fast delivery, and online-exclusive deals."),
    "Budget Conscious":  ("💡", "#f59e0b", "Lower income, price-sensitive. Spend less across all categories. Best engaged with discounts, value bundles, and promotional campaigns."),
    "Passive Segment":   ("😴", "#94a3b8", "Low engagement across all channels. Infrequent purchases, low recency. Need re-activation through win-back campaigns and personalised reminders."),
}

# ====================== SCALED CENTROIDS ======================
for col in pred_features:
    df[col + "_scaled"] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

centroids = {}
for model_col in ["kmeans_cluster", "SOM_Cluster", "DBSCAN_Cluster"]:
    scaled_cols = [f + "_scaled" for f in pred_features]
    centroids[model_col] = df.groupby(model_col)[scaled_cols].mean()

# ====================== MODEL VALIDATION SCORES ======================
model_stats = {
    "kmeans_cluster": {
        "name": "KMeans", "silhouette": 0.1940,
        "note": "Best silhouette score — clearest, most balanced clusters."
    },
    "SOM_Cluster": {
        "name": "SOM", "silhouette": -0.2655,
        "note": "Negative score — 179 micro-clusters overlap heavily. SOM captures topology, not separation."
    },
    "DBSCAN_Cluster": {
        "name": "DBSCAN", "silhouette": 0.0514,
        "note": f"{int((df['DBSCAN_Cluster'] == -1).sum())} customers (33.3%) flagged as outliers/noise."
    },
}

# ====================== AXIS OPTIONS ======================
axis_options = [
    {"label": "Income",             "value": "Income"},
    {"label": "Recency (days)",     "value": "Recency"},
    {"label": "Wines Spent",        "value": "MntWines"},
    {"label": "Meat Spent",         "value": "MntMeatProducts"},
    {"label": "Fish Spent",         "value": "MntFishProducts"},
    {"label": "Web Purchases",      "value": "NumWebPurchases"},
    {"label": "Store Purchases",    "value": "NumStorePurchases"},
    {"label": "Catalog Purchases",  "value": "NumCatalogPurchases"},
]

# ====================== INCOME SLIDER MARKS ======================
z_min = round(df["Income"].min(), 1)
z_max = round(df["Income"].max(), 1)
income_marks = {}
for z in [z_min, z_min / 2, 0.0, z_max / 2, z_max]:
    raw = z_to_raw_income(z)
    income_marks[round(z, 2)] = f"R{raw/1000:.0f}k"

# ====================== DASH APP ======================
app = Dash(__name__, external_stylesheets=["assets/style.css"])

app.layout = html.Div([
    html.H1("Customer Segmentation Dashboard", className="header"),

    # ── KPI CARDS ──────────────────────────────────────────────────────────
    html.Div([
        html.Div([html.H4("Total Customers"),  html.H2(f"{len(df):,}",                     className="kpi-number")], className="card kpi-card"),
        html.Div([html.H4("KMeans Clusters"),  html.H2(f"{df['kmeans_cluster'].nunique()}", className="kpi-number")], className="card kpi-card"),
        html.Div([html.H4("SOM Clusters"),     html.H2(f"{df['SOM_Cluster'].nunique()}",    className="kpi-number")], className="card kpi-card"),
        html.Div([html.H4("DBSCAN Clusters"),  html.H2(f"{df['DBSCAN_Cluster'].nunique()}", className="kpi-number")], className="card kpi-card"),
    ], className="kpi-container"),

    # ── SEGMENT PROFILE CARDS ───────────────────────────────────────────────
    html.Div([
        html.H3("Customer Segment Profiles", className="card-title"),
        html.P("Each segment represents a distinct customer group identified across all three clustering models.",
               style={"color": "#64748b", "marginBottom": "20px"}),
        html.Div([
            html.Div([
                html.Div([
                    html.Span(icon, style={"fontSize": "1.8rem", "marginRight": "10px"}),
                    html.Span(seg,  style={"fontWeight": "700", "fontSize": "1.05rem", "color": color}),
                ], style={"display": "flex", "alignItems": "center", "marginBottom": "8px"}),
                html.P(desc, style={"color": "#475569", "fontSize": "0.9rem", "margin": "0"}),
            ], style={
                "background": "white", "border": f"2px solid {color}", "borderRadius": "12px",
                "padding": "16px", "flex": "1", "minWidth": "220px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.06)"
            })
            for seg, (icon, color, desc) in segment_descriptions.items()
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ], className="card", style={"marginBottom": "30px"}),

    # ── MODEL SELECTOR ─────────────────────────────────────────────────────
    html.Div([
        html.Label("Select Clustering Model", className="label"),
        dcc.Dropdown(id="model_selector", options=[
            {"label": "KMeans", "value": "kmeans_cluster"},
            {"label": "SOM",    "value": "SOM_Cluster"},
            {"label": "DBSCAN", "value": "DBSCAN_Cluster"},
        ], value="kmeans_cluster", clearable=False, className="dropdown"),
    ], className="selector-container"),

    # ── INCOME FILTER (shows real Rand values) ─────────────────────────────
    html.Div([
        html.H4("Filter by Income", className="label"),
        html.P("Drag to show only customers up to the selected income level.",
               style={"color": "#64748b", "marginBottom": "10px", "fontSize": "0.9rem"}),
        dcc.Slider(id="income_filter", min=z_min, max=z_max, step=0.05, value=z_max,
                   marks=income_marks,
                   tooltip={"placement": "bottom", "always_visible": True}),
        html.Div(id="income_slider_display",
                 style={"textAlign": "right", "color": "#334155", "fontWeight": "600",
                        "marginTop": "6px", "fontSize": "0.95rem"}),
    ], className="filter-container"),

    # ── SCATTER AXIS SELECTORS ─────────────────────────────────────────────
    html.Div([
        html.Div([
            html.Label("X Axis", className="label"),
            dcc.Dropdown(id="x_axis", options=axis_options, value="Income",
                         clearable=False, className="dropdown"),
        ], style={"flex": "1"}),
        html.Div([
            html.Label("Y Axis", className="label"),
            dcc.Dropdown(id="y_axis", options=axis_options, value="MntWines",
                         clearable=False, className="dropdown"),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "20px", "marginBottom": "25px"}),

    # ── CUSTOMER PROFILE INPUT ─────────────────────────────────────────────
    html.Div([
        html.H3("Enter Customer Profile (RAW values)", className="card-title"),
        html.P("Income in Rands • Mnt* = total amount spent • Recency = days since last purchase • Purchases = count. The app automatically standardises your input before predicting.", className="warning"),

        html.Div([
            html.Div([html.Label("Income (R)"),          dcc.Input(id="income",  type="number", placeholder="e.g. 50000", className="input", style={"height":"52px"})], className="input-group"),
            html.Div([html.Label("Recency (days)"),      dcc.Input(id="recency", type="number", placeholder="e.g. 50",    className="input", style={"height":"52px"})], className="input-group"),
            html.Div([html.Label("MntWines (R)"),        dcc.Input(id="wine",    type="number", placeholder="e.g. 300",   className="input", style={"height":"52px"})], className="input-group"),
        ], className="input-row"),
        html.Div([
            html.Div([html.Label("MntMeatProducts (R)"), dcc.Input(id="meat",    type="number", placeholder="e.g. 200",   className="input", style={"height":"52px"})], className="input-group"),
            html.Div([html.Label("MntFishProducts (R)"), dcc.Input(id="fish",    type="number", placeholder="e.g. 80",    className="input", style={"height":"52px"})], className="input-group"),
            html.Div([html.Label("NumWebPurchases"),     dcc.Input(id="web",     type="number", placeholder="e.g. 4",     className="input", style={"height":"52px"})], className="input-group"),
        ], className="input-row"),
        html.Div([
            html.Div([html.Label("NumStorePurchases"),   dcc.Input(id="store",   type="number", placeholder="e.g. 8",     className="input", style={"height":"52px"})], className="input-group"),
            html.Div([html.Label("NumCatalogPurchases"), dcc.Input(id="catalog", type="number", placeholder="e.g. 3",     className="input", style={"height":"52px"})], className="input-group"),
            html.Div([html.Button("Predict Cluster", id="predict_btn", className="btn btn-primary",
                                  style={"marginTop": "28px"})], className="input-group"),
        ], className="input-row"),

        html.Button("Reset All Fields", id="reset_btn", className="btn btn-secondary"),
        html.Div(id="prediction_output", className="prediction-box"),
    ], className="card input-card"),

    # ── MODEL COMPARISON TABLE ──────────────────────────────────────────────
    html.Div([
        html.H3("Model Comparison", className="card-title"),
        html.P("Side-by-side summary of all three clustering models.",
               style={"color": "#64748b", "marginBottom": "16px"}),
        html.Table([
            html.Thead(html.Tr([
                html.Th("Model",            style={"textAlign":"left",   "padding":"10px 16px","background":"#1e293b","color":"white","borderRadius":"8px 0 0 0"}),
                html.Th("Clusters",         style={"textAlign":"center", "padding":"10px 16px","background":"#1e293b","color":"white"}),
                html.Th("Silhouette Score", style={"textAlign":"center", "padding":"10px 16px","background":"#1e293b","color":"white"}),
                html.Th("Noise / Outliers", style={"textAlign":"center", "padding":"10px 16px","background":"#1e293b","color":"white"}),
                html.Th("Notes",            style={"textAlign":"left",   "padding":"10px 16px","background":"#1e293b","color":"white","borderRadius":"0 8px 0 0"}),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td("KMeans", style={"padding":"10px 16px","fontWeight":"600"}),
                    html.Td("4",      style={"textAlign":"center","padding":"10px 16px"}),
                    html.Td("0.194",  style={"textAlign":"center","padding":"10px 16px","color":"#16a34a","fontWeight":"700"}),
                    html.Td("None",   style={"textAlign":"center","padding":"10px 16px"}),
                    html.Td("Best defined clusters. Clear segment separation.", style={"padding":"10px 16px","color":"#475569"}),
                ]),
                html.Tr([
                    html.Td("SOM",    style={"padding":"10px 16px","fontWeight":"600"}),
                    html.Td("179",    style={"textAlign":"center","padding":"10px 16px"}),
                    html.Td("-0.266", style={"textAlign":"center","padding":"10px 16px","color":"#dc2626","fontWeight":"700"}),
                    html.Td("None",   style={"textAlign":"center","padding":"10px 16px"}),
                    html.Td("Captures topological relationships; many overlapping micro-clusters.", style={"padding":"10px 16px","color":"#475569"}),
                ], style={"background":"#f8fafc"}),
                html.Tr([
                    html.Td("DBSCAN", style={"padding":"10px 16px","fontWeight":"600"}),
                    html.Td("4",      style={"textAlign":"center","padding":"10px 16px"}),
                    html.Td("0.051",  style={"textAlign":"center","padding":"10px 16px","color":"#d97706","fontWeight":"700"}),
                    html.Td("745 (33.3%)", style={"textAlign":"center","padding":"10px 16px","color":"#dc2626"}),
                    html.Td("4 clusters total: 3 meaningful groups + Cluster -1 (outliers/noise). 33% of customers flagged as outliers.", style={"padding":"10px 16px","color":"#475569"}),
                ]),
            ])
        ], style={"width":"100%","borderCollapse":"collapse","borderRadius":"10px","overflow":"hidden",
                  "border":"1px solid #e2e8f0","fontSize":"0.95rem"}),
    ], className="card", style={"marginTop":"30px"}),

    # ── SILHOUETTE SCORE CARD ───────────────────────────────────────────────
    html.Div(id="silhouette_card", className="card", style={"marginTop":"20px"}),

    # ── CHARTS ─────────────────────────────────────────────────────────────
    dcc.Graph(id="graph",     className="graph-card"),
    dcc.Graph(id="pie-chart", className="graph-card"),
    dcc.Graph(id="bar-chart", className="graph-card"),

    html.Div(id="insights", className="card insights-card"),

], className="main-container")


# ====================== CALLBACKS ======================

@app.callback(
    Output("income_slider_display", "children"),
    Input("income_filter", "value")
)
def update_income_display(z_val):
    raw = z_to_raw_income(z_val)
    return f"Showing customers with income up to R{raw:,.0f}"


@app.callback(
    [Output("graph", "figure"), Output("pie-chart", "figure"),
     Output("bar-chart", "figure"), Output("insights", "children"),
     Output("silhouette_card", "children")],
    [Input("model_selector", "value"), Input("income_filter", "value"),
     Input("x_axis", "value"),         Input("y_axis", "value")]
)
def update_graphs(model, income_filter, x_col, y_col):
    filtered_df = df[df["Income"] <= income_filter]

    x_label = next((o["label"] for o in axis_options if o["value"] == x_col), x_col)
    y_label = next((o["label"] for o in axis_options if o["value"] == y_col), y_col)

    # ── Scatter ──
    if model == "SOM_Cluster":
        scatter_fig = px.scatter(
            filtered_df, x=x_col, y=y_col, color=model,
            title=f"SOM Clustering — {x_label} vs {y_label}",
            color_continuous_scale=px.colors.sequential.Viridis,
            labels={x_col: x_label, y_col: y_label},
        )
    else:
        scatter_fig = px.scatter(
            filtered_df, x=x_col, y=y_col, color=model,
            title=f"{model.replace('_cluster','').replace('_Cluster','').title()} Clustering — {x_label} vs {y_label}",
            color_discrete_sequence=px.colors.qualitative.Set1,
            labels={x_col: x_label, y_col: y_label},
        )
    scatter_fig.update_layout(height=500, template="plotly_white", margin=dict(l=40, r=40, t=60, b=40))

    # ── Pie chart ──
    pie_data = df.copy()
    pie_data["Cluster_Label"] = pie_data[model].apply(lambda c: make_label(model, c))
    pie_fig = px.pie(
        pie_data, names="Cluster_Label",
        title=f"Cluster Distribution — {model.replace('_',' ').title()}",
        color_discrete_sequence=px.colors.qualitative.Set1,
    )
    pie_fig.update_layout(height=500, margin=dict(l=40, r=40, t=60, b=40))

    # ── Bar chart — average spending per cluster ──
    spending_cols = ["MntWines", "MntMeatProducts", "MntFishProducts"]
    bar_rows = []
    for cluster_id, group in df.groupby(model):
        label = make_label(model, cluster_id)
        for col in spending_cols:
            z_mean    = group[col].mean()
            raw_approx = max(0, np.expm1(z_mean * scaler_sigma[col] + scaler_mu[col]))
            bar_rows.append({
                "Cluster":      label,
                "Category":     col.replace("Mnt","").replace("Products",""),
                "Avg Spend (R)": round(raw_approx, 1)
            })
    bar_df  = pd.DataFrame(bar_rows)
    bar_fig = px.bar(
        bar_df, x="Cluster", y="Avg Spend (R)", color="Category", barmode="group",
        title=f"Average Spending per Cluster — {model.replace('_',' ').title()}",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    bar_fig.update_layout(height=450, template="plotly_white",
                          margin=dict(l=40, r=40, t=60, b=100), xaxis_tickangle=-20)

    # ── Insight ──
    insight = {
        "kmeans_cluster":  "KMeans creates clear, balanced segments based on spending behaviour.",
        "SOM_Cluster":     "SOM shows deeper topological relationships between customers.",
        "DBSCAN_Cluster":  "DBSCAN finds dense clusters and flags outliers (cluster -1 = noise/outliers — customers who don't fit any cluster)."
    }[model]

    # ── Silhouette card ──
    stats  = model_stats[model]
    score  = stats["silhouette"]
    s_color = "#16a34a" if score > 0.15 else ("#d97706" if score > 0 else "#dc2626")
    s_label = "Good" if score > 0.15 else ("Weak" if score > 0 else "Negative (overlapping clusters)")
    sil_card = html.Div([
        html.H4(f"Validation — {stats['name']} Silhouette Score", style={"marginBottom":"10px"}),
        html.Div([
            html.Span(f"{score:.4f}", style={"fontSize":"2.2rem","fontWeight":"700","color":s_color}),
            html.Span(f"  —  {s_label}", style={"fontSize":"1rem","color":"#64748b","marginLeft":"12px"}),
        ]),
        html.P(stats["note"], style={"marginTop":"10px","color":"#475569"}),
        html.P("Silhouette score ranges from -1 to +1. Higher means clusters are well separated and compact.",
               style={"fontSize":"0.85rem","color":"#94a3b8","marginTop":"6px"}),
    ])

    return scatter_fig, pie_fig, bar_fig, html.Div([html.H4("Model Insight"), html.P(insight)]), sil_card


@app.callback(
    Output("prediction_output", "children"),
    Input("predict_btn", "n_clicks"),
    State("model_selector", "value"),
    State("income", "value"), State("recency", "value"),
    State("wine", "value"),   State("meat", "value"), State("fish", "value"),
    State("web", "value"),    State("store", "value"), State("catalog", "value"),
)
def predict_cluster(n_clicks, model, income, recency, wine, meat, fish, web, store, catalog):
    if n_clicks is None:
        return ""
    if None in (income, recency, wine, meat, fish, web, store, catalog):
        return html.Div("Please fill in ALL fields", className="error-msg")

    raw_values = {
        "Income": income, "Recency": recency, "MntWines": wine,
        "MntMeatProducts": meat, "MntFishProducts": fish,
        "NumWebPurchases": web, "NumStorePurchases": store,
        "NumCatalogPurchases": catalog
    }

    # raw → z-score (log1p for Mnt* cols) → min-max into centroid space
    zscores = {col: raw_to_zscore(col, raw_values[col]) for col in pred_features}
    input_scaled = pd.Series({
        f"{col}_scaled": (zscores[col] - df[col].min()) / (df[col].max() - df[col].min())
        for col in pred_features
    })

    distances  = ((centroids[model] - input_scaled) ** 2).sum(axis=1) ** 0.5
    predicted  = distances.idxmin()
    cluster_rows = df[df[model] == predicted]
    segment    = cluster_rows["Segment"].mode().iloc[0] if not cluster_rows.empty else "Unknown"

    avgs = cluster_rows[pred_features].mean().round(2)
    table_rows = [html.Tr([
        html.Td(col,        style={"padding":"6px 12px"}),
        html.Td(avgs[col],  style={"padding":"6px 12px","textAlign":"right"}),
    ]) for col in pred_features]
    avg_table = html.Table([
        html.Thead(html.Tr([
            html.Th("Feature",                   style={"textAlign":"left",  "padding":"8px 12px","background":"#f1f5f9"}),
            html.Th("Cluster Average (z-score)", style={"textAlign":"right", "padding":"8px 12px","background":"#f1f5f9"}),
        ])),
        html.Tbody(table_rows)
    ], style={"width":"100%","marginTop":"15px","borderCollapse":"collapse",
              "fontSize":"0.95rem","border":"1px solid #e2e8f0","borderRadius":"8px"})

    rec_text = marketing_recs.get(segment, "No specific recommendation available.")
    rec = "OUTLIER detected – investigate unusual behaviour." if predicted == -1 else \
          f"Matches the {segment} segment. Good target for loyalty rewards or cross-selling."

    return html.Div([
        html.H4(f"Predicted Cluster: {predicted}"),
        html.P(f"Segment: {segment}", style={"fontWeight":"600","fontSize":"1.1rem"}),
        html.P(rec, className="recommendation"),
        html.H5("Marketing Recommendation", style={"marginTop":"20px","marginBottom":"8px"}),
        html.P(rec_text, style={"fontStyle":"italic","color":"#1e293b"}),
        html.H5("Average Values in this Cluster", style={"marginTop":"25px","marginBottom":"8px"}),
        avg_table
    ], className="prediction-result")


@app.callback(
    [Output("income","value"), Output("recency","value"), Output("wine","value"),
     Output("meat","value"),   Output("fish","value"),    Output("web","value"),
     Output("store","value"),  Output("catalog","value")],
    Input("reset_btn", "n_clicks")
)
def reset_inputs(n_clicks):
    if n_clicks is None:
        return [None] * 8
    return [None] * 8


if __name__ == "__main__":
    app.run(debug=False)