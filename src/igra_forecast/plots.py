from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = {
    "truth": "#3F4A56",
    "prediction": "#B7795F",
    "hist": "#86A9C8",
    "rmse": "#7FA6C7",
    "mae": "#A8C3B5",
    "accent": "#C9855A",
    "green": "#8FB9A8",
    "purple": "#A997C9",
    "gray": "#6B7280",
    "grid": "#E8ECEF",
}


def set_journal_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "Arial",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.35,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def plot_predictions(predictions: pd.DataFrame, figures_dir: Path, cfg: dict) -> None:
    set_journal_style()
    dpi = int(cfg.get("plots", {}).get("dpi", 600))
    max_points = int(cfg.get("plots", {}).get("max_points", 500))
    for target in sorted(predictions["target"].unique()):
        for horizon in sorted(predictions["horizon"].unique()):
            subset = predictions[(predictions["target"] == target) & (predictions["horizon"] == horizon)].copy()
            subset = subset.sort_values("target_time")
            if len(subset) > max_points:
                subset = subset.iloc[-max_points:]
            fig, ax = plt.subplots(figsize=(6.6, 2.6))
            ax.plot(
                pd.to_datetime(subset["target_time"]),
                subset["y_true"],
                color=PALETTE["truth"],
                linewidth=1.35,
                alpha=0.86,
                label="Observed",
            )
            ax.plot(
                pd.to_datetime(subset["target_time"]),
                subset["y_pred"],
                color=PALETTE["prediction"],
                linewidth=1.25,
                alpha=0.94,
                label="Predicted",
            )
            ax.set_title(f"{target}, lead {horizon} day")
            ax.set_xlabel("Date")
            ax.set_ylabel(target)
            ax.legend(frameon=False, ncol=2, loc="upper right", handlelength=2.2, columnspacing=1.3)
            _clean_axis(ax)
            ax.grid(True, color=PALETTE["grid"], linewidth=0.55, alpha=0.7)
            fig.autofmt_xdate(rotation=20)
            fig.savefig(figures_dir / f"prediction_{target}_lead{horizon}.png", dpi=dpi)
            plt.close(fig)


def plot_error_distribution(predictions: pd.DataFrame, figures_dir: Path, cfg: dict) -> None:
    set_journal_style()
    dpi = int(cfg.get("plots", {}).get("dpi", 600))
    targets = sorted(predictions["target"].unique())
    fig_width = max(2.75 * len(targets), 6.6)
    fig, axes = plt.subplots(
        1,
        len(targets),
        figsize=(fig_width, 2.55),
        squeeze=False,
        constrained_layout=True,
        gridspec_kw={"wspace": 0.10},
    )
    for idx, (ax, target) in enumerate(zip(axes[0], targets)):
        subset = predictions[predictions["target"] == target]
        sns.histplot(
            subset["error"],
            bins=36,
            kde=True,
            color=PALETTE["hist"],
            ax=ax,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.72,
        )
        ax.axvline(0, color=PALETTE["accent"], linewidth=1.0, linestyle="--", zorder=3)
        ax.set_title(target, pad=6)
        ax.set_xlabel("Prediction error", labelpad=5)
        if idx == 0:
            ax.set_ylabel("Count", labelpad=5)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", pad=2)
        _clean_axis(ax)
        ax.grid(True, axis="y", alpha=0.28)
    fig.savefig(figures_dir / "error_distribution.png", dpi=dpi)
    plt.close(fig)


def plot_metric_summary(metrics: pd.DataFrame, figures_dir: Path, cfg: dict) -> None:
    set_journal_style()
    dpi = int(cfg.get("plots", {}).get("dpi", 600))
    subset = metrics[(metrics["scope"] == "target_horizon") & (metrics["metric"].isin(["rmse", "mae"]))]
    if subset.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.15), constrained_layout=True)
    plot_df = subset.copy()
    plot_df["label"] = plot_df["target"] + " | lead " + plot_df["horizon"].astype(str)
    sns.barplot(
        data=plot_df,
        x="label",
        y="value",
        hue="metric",
        palette=[PALETTE["rmse"], PALETTE["mae"]],
        edgecolor="white",
        linewidth=0.6,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Metric value")
    ax.set_title("Forecast error summary")
    ax.tick_params(axis="x", rotation=32, labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend(frameon=False, title="", loc="upper right", bbox_to_anchor=(0.995, 0.99))
    ax.margins(x=0.015)
    _clean_axis(ax)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.75)
    fig.savefig(figures_dir / "metrics_summary.png", dpi=dpi)
    plt.close(fig)


def plot_loss_curve(training_history: pd.DataFrame | None, figures_dir: Path, cfg: dict, model_name: str) -> None:
    set_journal_style()
    dpi = int(cfg.get("plots", {}).get("dpi", 600))
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    if training_history is None or training_history.empty:
        ax.text(
            0.5,
            0.5,
            "No iterative training loss\nfor this baseline model",
            ha="center",
            va="center",
            color=PALETTE["gray"],
            transform=ax.transAxes,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.plot(training_history["epoch"], training_history["train_loss"], color=PALETTE["prediction"], label="Train")
        if "val_loss" in training_history:
            ax.plot(training_history["epoch"], training_history["val_loss"], color=PALETTE["accent"], label="Validation")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(frameon=False)
    ax.set_title(f"Training curve: {model_name}")
    _clean_axis(ax)
    fig.savefig(figures_dir / "loss_curve.png", dpi=dpi)
    plt.close(fig)


def _clean_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#C9CDD2")
    ax.spines["bottom"].set_color("#C9CDD2")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, axis="y", color=PALETTE["grid"], linewidth=0.55, alpha=0.7)
