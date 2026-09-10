from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from _bootstrap import PROJECT_ROOT  # noqa: F401
from igra_forecast.revision2_data import sha256_file, write_json


COLORS = {
    "era5": "#9AA6B2", "hgb": "#1B3D6E", "dry": "#4C78A8", "wet": "#E07A5F",
    "cross": "#7A5195", "total": "#222222", "positive": "#B14A48", "negative": "#2A6F97",
}


def read_config(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def require(path: str | Path) -> Path:
    result = Path(path)
    if not result.is_file():
        raise FileNotFoundError(f"Figure input is missing: {result}")
    return result


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9.0, "axes.labelsize": 9.5, "axes.titlesize": 10.5,
        "legend.fontsize": 8.0, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.linewidth": 0.9, "lines.linewidth": 1.6, "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": False,
    })


def finish(fig: plt.Figure, output: Path, name: str) -> list[Path]:
    png, pdf = output / f"{name}.png", output / f"{name}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def open_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=3)
    ax.grid(axis="y", color="#E5E5E5", lw=0.65, ls="--", zorder=0)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.10, 1.04, label, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold", clip_on=False)


def copy_data(frame: pd.DataFrame, data_root: Path, name: str) -> Path:
    path = data_root / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def figure_workflow(output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    nodes = pd.DataFrame([
        [0.10, 0.68, "ERA5 + IGRA\n2024–2025"], [0.36, 0.68, "Dry/wet error\ndecomposition"],
        [0.62, 0.68, "Frozen HGB\nstatistical probe"], [0.88, 0.68, "Internal spatial, temporal,\nand space-time tests"],
        [0.24, 0.25, "RAPSODI\nregional diagnostic"], [0.50, 0.25, "COSMIC-2 MSL height\ncross-platform diagnostic"],
        [0.76, 0.25, "Calibration transfer and\napplicability limits"],
    ], columns=["x", "y", "label"])
    edges = pd.DataFrame([[0, 1], [1, 2], [2, 3], [2, 4], [2, 5], [3, 6], [4, 6], [5, 6]], columns=["source", "target"])
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    for edge in edges.itertuples(index=False):
        a, b = nodes.iloc[edge.source], nodes.iloc[edge.target]
        ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y), arrowprops=dict(arrowstyle="->", color="#6B7280", lw=1.3))
    for index, node in nodes.iterrows():
        color = "#DCE8F5" if index < 4 else "#F3E7DF"
        ax.text(node.x, node.y, node.label, ha="center", va="center", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.48", fc=color, ec="#34495E", lw=1.0))
    ax.text(0.5, 0.94, "Evidence chain for physical diagnosis and generalization limits", ha="center", va="center", fontsize=11.5, fontweight="bold")
    return finish(fig, output, "fig01_evidence_workflow"), [copy_data(nodes, data_root, "fig01_nodes"), copy_data(edges, data_root, "fig01_edges")]


def figure_observations(config: dict[str, object], output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    stations = pd.read_csv(require("data/revision2/manifests/station_split.csv"))
    cosmic = pd.read_parquet(require("data/revision2/processed/cosmic2_atmprf_height_profiles.parquet"), columns=["profile_id", "time", "latitude", "longitude"])
    cosmic = cosmic.drop_duplicates(["profile_id", "time"]).iloc[::20].copy()
    rapsodi_path = Path(config["external_statistics"]["rapsodi_predictions"])
    rapsodi = pd.read_parquet(require(rapsodi_path), columns=["station_id", "latitude", "longitude"]).drop_duplicates()
    projection = ccrs.Robinson()
    fig = plt.figure(figsize=(7.2, 3.7))
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_global(); ax.add_feature(cfeature.LAND, facecolor="#F1EFEA", zorder=0); ax.add_feature(cfeature.OCEAN, facecolor="#EAF2F8", zorder=0)
    ax.coastlines(lw=0.55, color="#555555"); ax.gridlines(lw=0.35, color="#AAAAAA", alpha=0.6, linestyle=":")
    ax.scatter(cosmic.longitude, cosmic.latitude, s=1.3, c="#A6A6A6", alpha=0.25, transform=ccrs.PlateCarree(), label="COSMIC-2 (1/20 profiles)", rasterized=True)
    ax.scatter(stations.longitude, stations.latitude, s=9, c="#1B3D6E", alpha=0.78, edgecolors="white", linewidths=0.18, transform=ccrs.PlateCarree(), label="IGRA")
    ax.scatter(rapsodi.longitude, rapsodi.latitude, s=30, marker="*", c="#C44E52", edgecolors="white", linewidths=0.4, transform=ccrs.PlateCarree(), label="RAPSODI/INMG")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)
    ax.set_title("Observational systems used for internal and cross-platform evaluation", pad=8)
    data = [copy_data(stations, data_root, "fig02_igra_stations"), copy_data(cosmic, data_root, "fig02_cosmic_sample"), copy_data(rapsodi, data_root, "fig02_rapsodi")]
    return finish(fig, output, "fig02_observing_systems"), data


def figure_error_budget(config: dict[str, object], output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    budget = pd.read_csv(require(Path(config["paper_output_root"]) / "physical_error_budget.csv"))
    budget = budget.query("grouping == 'pressure_level'").copy()
    budget["group_numeric"] = pd.to_numeric(budget["group"])
    budget = budget.sort_values("group_numeric", ascending=False)
    levels = budget["group"].astype(int).astype(str).tolist()
    series = [("Dry", "hgb_minus_era5_dry_mse", COLORS["dry"]), ("Wet", "hgb_minus_era5_wet_mse", COLORS["wet"]),
              ("Cross", "hgb_minus_era5_cross_term", COLORS["cross"]), ("Total", "hgb_minus_era5_total_mse", COLORS["total"])]
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    x, width = np.arange(len(levels)), 0.19
    for offset, (label, column, color) in enumerate(series):
        ax.bar(x + (offset - 1.5) * width, budget[column], width, color=color, label=label, edgecolor="white", zorder=2)
    ax.axhline(0, color="#333333", lw=0.9); ax.set_xticks(x, levels); ax.set_xlabel("Pressure level (hPa)")
    ax.set_ylabel(r"Change in MSE term (HGB $-$ ERA5; N-units$^2$)")
    ax.legend(ncol=4, frameon=False, loc="upper center"); open_axes(ax); ax.set_title("Pressure-level physical error budget")
    return finish(fig, output, "fig03_physical_error_budget"), [copy_data(budget, data_root, "fig03_physical_error_budget")]


def figure_generalization(config: dict[str, object], output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    internal = pd.read_csv(require(Path(config["paper_output_root"]) / "internal_generalization_summary.csv"))
    data = internal.query("model == 'hgb'").copy()
    labels = {"spatial_station_disjoint": "Spatial station holdout", "temporal_holdout": "Future-time holdout", "space_time_holdout": "Space–time holdout"}
    data["label"] = data["protocol"].map(labels)
    fig, ax = plt.subplots(figsize=(5.2, 3.2), constrained_layout=True)
    y = np.arange(len(data))[::-1]
    values = data["station_mean_rmse_difference_vs_era5"].to_numpy(float)
    lower = values - data["station_ci_lower"].to_numpy(float); upper = data["station_ci_upper"].to_numpy(float) - values
    ax.errorbar(values, y, xerr=np.vstack([lower, upper]), fmt="o", color=COLORS["hgb"], ecolor="#54789D", capsize=3, ms=5)
    ax.axvline(0, color="#555555", ls="--", lw=1); ax.set_yticks(y, data["label"]); ax.set_xlabel("Station-mean RMSE difference (HGB − ERA5; N-units)")
    open_axes(ax); ax.grid(axis="x", color="#E5E5E5", lw=0.65, ls="--"); ax.grid(axis="y", visible=False)
    ax.set_title("Internal generalization with station-cluster 95% intervals")
    return finish(fig, output, "fig04_internal_generalization"), [copy_data(data, data_root, "fig04_internal_generalization")]


def figure_blocks(config: dict[str, object], output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    runs = pd.read_csv(require(Path(config["confirmatory_root"]) / "spatial_leave_block_out" / "summary" / "spatial_block_runs.csv"))
    data = runs.groupby(["period", "block"], as_index=False).agg(hgb_minus_era5_rmse=("hgb_minus_era5_rmse", "first"), held_out_stations=("held_out_stations", "first"), level_samples=("level_samples", "first"))
    pivot = data.pivot(index="block", columns="period", values="hgb_minus_era5_rmse").sort_values("future")
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    y = np.arange(len(pivot))
    for index in range(len(pivot)):
        ax.plot([pivot.iloc[index]["concurrent"], pivot.iloc[index]["future"]], [y[index], y[index]], color="#C7CDD4", lw=1.0)
    ax.scatter(pivot["concurrent"], y, color="#6C8EBF", s=24, label="Concurrent 2024", zorder=3)
    ax.scatter(pivot["future"], y, color="#C65D57", marker="s", s=24, label="Future 2025", zorder=3)
    ax.axvline(0, color="#444444", ls="--", lw=1); ax.set_yticks(y, pivot.index); ax.set_xlabel("Block RMSE difference (HGB − ERA5; N-units)")
    ax.legend(frameon=False, ncol=2, loc="lower right"); open_axes(ax); ax.grid(axis="x", color="#E5E5E5", lw=0.65, ls="--"); ax.grid(axis="y", visible=False)
    ax.set_title("Regional heterogeneity in 20 spatial-block pressure tests")
    return finish(fig, output, "fig05_spatial_block_heterogeneity"), [copy_data(data, data_root, "fig05_spatial_blocks")]


def figure_probability(config: dict[str, object], output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    data = pd.read_csv(require(Path(config["paper_output_root"]) / "probability_transfer_summary.csv"))
    data = data.loc[~((data["domain"] == "COSMIC2") & (data["coordinate"] == "dry_pressure_sensitivity"))].copy()
    label_map = {
        "hgb_prob_global": "IGRA\nglobal SD", "hgb_prob_hetero_diag": "IGRA\nheteroscedastic",
        "hgb_prob_hetero_structured": "IGRA\nstructured", "frozen_structured_HGB": "",
    }
    data["label"] = [
        label_map.get(row.model, "") if str(row.domain).startswith("IGRA")
        else ("RAPSODI" if str(row.domain).startswith("RAPSODI") else "COSMIC-2\nMSL height")
        for row in data.itertuples(index=False)
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.5), constrained_layout=True)
    colors = ["#3B6BB5", "#D99B62", "#B14A48"]
    bars = ax.bar(np.arange(len(data)), data["marginal_coverage_90"], color=colors, width=0.58, edgecolor="white", zorder=2)
    ax.axhline(0.90, color="#333333", ls="--", lw=1.1, label="Nominal 90%")
    ax.set_xticks(np.arange(len(data)), data["label"]); ax.set_ylim(0, 1.0); ax.set_ylabel("Empirical marginal coverage")
    for bar, value in zip(bars, data["marginal_coverage_90"], strict=True):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
    ax.legend(frameon=False, loc="lower left"); open_axes(ax); ax.set_title("Predictive-interval calibration does not transfer across platforms")
    return finish(fig, output, "fig06_probability_transfer"), [copy_data(data, data_root, "fig06_probability_transfer")]


def figure_height_performance(statistics: Path, output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    level = pd.read_csv(require(statistics / "cosmic_height_level_metrics.csv"))
    block = pd.read_csv(require(statistics / "cosmic_height_block_bootstrap.csv")).query("grouping == 'height_level'")
    data = level.merge(block[["height_km", "mean_difference", "ci_lower", "ci_upper"]], on="height_km", how="left", validate="one_to_one")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    axes[0].plot(data.height_km, data.era5_rmse, "o-", color=COLORS["era5"], label="ERA5", ms=4)
    axes[0].plot(data.height_km, data.hgb_rmse, "s-", color=COLORS["hgb"], label="HGB", ms=3.5)
    axes[0].set_xlabel("MSL geometric height (km)"); axes[0].set_ylabel("RMSE (N-units)"); axes[0].legend(frameon=False); open_axes(axes[0]); panel_label(axes[0], "(a)")
    axes[1].fill_between(data.height_km, data.ci_lower, data.ci_upper, color=COLORS["hgb"], alpha=0.15)
    axes[1].plot(data.height_km, data.mean_difference, "o-", color=COLORS["hgb"], ms=4)
    axes[1].axhline(0, color="#444444", ls="--", lw=1); axes[1].set_xlabel("MSL geometric height (km)"); axes[1].set_ylabel("Archive-month mean RMSE difference\n(HGB − ERA5; N-units)"); open_axes(axes[1]); panel_label(axes[1], "(b)")
    fig.suptitle("COSMIC-2 geometric-height cross-platform comparison", fontsize=11)
    return finish(fig, output, "fig07_cosmic_height_performance"), [copy_data(data, data_root, "fig07_cosmic_height_performance")]


def figure_height_probability(statistics: Path, output: Path, data_root: Path) -> tuple[list[Path], list[Path]]:
    data = pd.read_csv(require(statistics / "cosmic_height_level_metrics.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.9), constrained_layout=True)
    axes[0].plot(data.height_km, data.coverage_90, "o-", color="#B14A48", ms=3.5); axes[0].axhline(.9, color="#444444", ls="--", lw=1)
    axes[0].set_ylim(.3, 1.0); axes[0].set_xlabel("Height (km)"); axes[0].set_ylabel("90% marginal coverage"); open_axes(axes[0]); panel_label(axes[0], "(a)")
    axes[1].plot(data.height_km, data.mean_correction, "o-", color="#2A6F97", ms=3.5); axes[1].axhline(0, color="#444444", ls="--", lw=1)
    axes[1].set_xlabel("Height (km)"); axes[1].set_ylabel("Mean correction (N-units)"); open_axes(axes[1]); panel_label(axes[1], "(b)")
    axes[2].plot(data.height_km, data.mean_predictive_sd, "o-", color="#7A5195", ms=3.5)
    axes[2].set_xlabel("Height (km)"); axes[2].set_ylabel("Mean predictive SD (N-units)"); open_axes(axes[2]); panel_label(axes[2], "(c)")
    fig.suptitle("Height-dependent predictive calibration and correction magnitude", fontsize=11)
    return finish(fig, output, "fig08_cosmic_height_probability"), [copy_data(data, data_root, "fig08_cosmic_height_probability")]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Revision 2.3 publication figures from frozen final statistics.")
    parser.add_argument("--config", default="configs/revision2/publication.yaml")
    parser.add_argument("--statistics-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = read_config(args.config)
    configure_style()
    final_root = Path(config["final_output_root"])
    statistics = Path(args.statistics_dir or final_root / "statistics")
    manifest = json.loads(require(statistics / "final_statistics_manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("smoke"):
        raise ValueError("Figures require complete formal final statistics")
    output = Path(args.output_dir or final_root / "figures" / "main")
    data_root = final_root / "figure_data"
    output.mkdir(parents=True, exist_ok=True); data_root.mkdir(parents=True, exist_ok=True)
    figure_files: list[Path] = []; data_files: list[Path] = []
    for builder, arguments in [
        (figure_workflow, (output, data_root)), (figure_observations, (config, output, data_root)),
        (figure_error_budget, (config, output, data_root)), (figure_generalization, (config, output, data_root)),
        (figure_blocks, (config, output, data_root)), (figure_probability, (config, output, data_root)),
        (figure_height_performance, (statistics, output, data_root)), (figure_height_probability, (statistics, output, data_root)),
    ]:
        figures, data = builder(*arguments); figure_files.extend(figures); data_files.extend(data)
    write_json(final_root / "figures_manifest.json", {
        "complete": True, "style": "plot-from-data-derived publication serif style", "dpi": 300,
        "statistics_manifest_sha256": sha256_file(statistics / "final_statistics_manifest.json"),
        "figures": {str(path.relative_to(final_root)): sha256_file(path) for path in figure_files},
        "figure_data": {str(path.relative_to(final_root)): sha256_file(path) for path in data_files},
    })
    print(f"Publication figures: {output.resolve()}")


if __name__ == "__main__":
    main()
