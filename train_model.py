"""
Train the Battery Health Predictor model.

Loads the raw dataset, engineers a "Capacity Ratio" feature
(Full Charge Capacity / Design Capacity), trains and compares four
regression models, and saves the best one (model.pkl / scaler.pkl /
model_meta.json) for app.py to load.

Usage:
    python train_model.py path/to/battery_health_dataset.csv
"""
import json
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

RANDOM_STATE = 42
TARGET = "Battery Health"
ASSETS_DIR = "assets"

ACCENT = "#2563eb"
MUTED_BLUES = ["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.edgecolor": "#e2e8f0",
    "axes.labelcolor": "#334155",
    "text.color": "#0f172a",
    "xtick.color": "#64748b",
    "ytick.color": "#64748b",
    "axes.grid": True,
    "grid.color": "#e2e8f0",
    "grid.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style_axes(ax):
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.set_axisbelow(True)

FEATURES = [
    "Battery Age",
    "Daily Usage Hours",
    "Gaming User",
    "Design Capacity",
    "Cycle Count",
    "CPU Usage",
    "GPU Usage",
    "Power Consumption",
    "Average Temperature",
    "Full Charge Capacity",
    "Capacity Ratio",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Capacity Ratio"] = df["Full Charge Capacity"] / df["Design Capacity"] * 100
    return df


def save_comparison_chart(results, metric, title, xlabel, fname, ascending):
    ordered = sorted(results, key=lambda r: r[metric], reverse=not ascending)
    names = [r["model"] for r in ordered]
    values = [r[metric] for r in ordered]
    colors = MUTED_BLUES[: len(names)]

    fig, ax = plt.subplots(figsize=(11, 1.1 * len(names) + 1.3))
    bars = ax.barh(names, values, color=colors)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel(xlabel, fontsize=13)
    style_axes(ax)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + max(values) * 0.015,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}" if metric == "r2" else f"{val:.2f}",
            va="center", fontsize=12, color="#0f172a",
        )
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/{fname}", dpi=140)
    plt.close(fig)


def save_feature_importance_chart(importances, model_name, fname):
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(figsize=(11, 0.62 * len(names) + 1.3))
    ax.barh(names, values, color=ACCENT)
    ax.invert_yaxis()
    ax.set_title(f"Feature Importance — {model_name}", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Feature Importance", fontsize=13)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/{fname}", dpi=140)
    plt.close(fig)


def save_correlation_chart(df, features, target, fname):
    cols = features + [target]
    corr = df[cols].corr()

    cmap = plt.get_cmap("RdBu_r")
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(cols, fontsize=11)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            # luminance of the actual cell color decides readable text color
            r, g, b, _ = cmap((val + 1) / 2)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            color = "white" if luminance < 0.6 else "#0f172a"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    ax.set_title("Feature Correlation Heatmap", fontsize=18, fontweight="bold", pad=14)
    fig.colorbar(im, ax=ax, shrink=0.85)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/{fname}", dpi=140)
    plt.close(fig)


def save_actual_vs_pred_chart(y_test, y_pred, model_name, fname):
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(y_test, y_pred, color=ACCENT, alpha=0.55, s=40, edgecolors="white", linewidths=0.5)
    lo = min(y_test.min(), y_pred.min())
    hi = max(y_test.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", color="#94a3b8", label="Perfect prediction")
    ax.set_title(f"Actual vs Predicted — {model_name}", fontsize=18, fontweight="bold", pad=14)
    ax.set_xlabel("Actual Battery Health (%)", fontsize=13)
    ax.set_ylabel("Predicted Battery Health (%)", fontsize=13)
    ax.legend(frameon=True, fontsize=11)
    style_axes(ax)
    ax.grid(axis="both", visible=True)
    fig.tight_layout()
    fig.savefig(f"{ASSETS_DIR}/{fname}", dpi=140)
    plt.close(fig)


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "battery_health_dataset.csv"
    df = pd.read_csv(csv_path)
    df = engineer_features(df)

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=RANDOM_STATE
        ),
    }

    results = []
    fitted = {}
    for name, mdl in candidates.items():
        mdl.fit(X_train_s, y_train)
        fitted[name] = mdl
        pred = mdl.predict(X_test_s)
        mae = mean_absolute_error(y_test, pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        r2 = r2_score(y_test, pred)
        results.append({"model": name, "mae": mae, "rmse": rmse, "r2": r2})
        print(f"{name:20s}  MAE={mae:.4f}  RMSE={rmse:.4f}  R2={r2:.4f}")

    results.sort(key=lambda r: r["r2"], reverse=True)
    best_name = results[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}")

    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(FEATURES, best_model.feature_importances_.tolist()))
    else:
        # fall back to |coefficient| share for models without native importances
        coefs = np.abs(getattr(best_model, "coef_", np.zeros(len(FEATURES))))
        total = coefs.sum() or 1.0
        importances = dict(zip(FEATURES, (coefs / total).tolist()))

    feature_stats = {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
            "median": float(X[col].median()),
        }
        for col in FEATURES
    }

    best_result = next(r for r in results if r["model"] == best_name)
    meta = {
        "model_name": best_name,
        "features": FEATURES,
        "target": TARGET,
        "metrics": {
            "mae": best_result["mae"],
            "rmse": best_result["rmse"],
            "r2": best_result["r2"],
        },
        "feature_stats": feature_stats,
        "feature_importance": importances,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "all_models_compared": results,
    }

    joblib.dump(best_model, "model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    with open("model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\nSaved model.pkl, scaler.pkl, model_meta.json")

    best_pred = best_model.predict(X_test_s)
    save_comparison_chart(results, "r2", "Model Comparison — R² Score",
                           "R² Score (closer to 1 is better)", "chart_r2.png", ascending=False)
    save_comparison_chart(results, "rmse", "Model Comparison — RMSE",
                           "RMSE (lower is better)", "chart_rmse.png", ascending=True)
    save_feature_importance_chart(importances, best_name, "chart_feature_importance.png")
    save_correlation_chart(df, FEATURES, TARGET, "chart_correlation.png")
    save_actual_vs_pred_chart(y_test, best_pred, best_name, "chart_actual_vs_pred.png")
    print("Saved chart_*.png in assets/")


if __name__ == "__main__":
    main()
