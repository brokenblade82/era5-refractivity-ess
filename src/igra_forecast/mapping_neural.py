from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from igra_forecast.mapping_neural_data import MappingScalers, SpatialEpisodeDataset, build_profile_table
from igra_forecast.refractivity_mapping import MappingPrediction, ResidualMappingModel, add_mapping_features


@dataclass
class SRNPArchitecture:
    query_dim: int = 16
    context_dim: int = 19
    relative_dim: int = 6
    output_dim: int = 3
    d_model: int = 128
    num_heads: int = 4
    context_layers: int = 2
    cross_layers: int = 2
    dropout: float = 0.1
    nearest_context: int = 16
    use_hgb_prior: bool = True
    use_context: bool = True
    use_geometry: bool = True
    use_distance_gate: bool = True
    joint_levels: bool = True


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(query, context, context, key_padding_mask=key_padding_mask, need_weights=False)
        query = self.norm1(query + attended)
        return self.norm2(query + self.ffn(query))


class HGBSphericalResidualNeuralProcess(nn.Module):
    def __init__(self, architecture: SRNPArchitecture):
        super().__init__()
        self.architecture = architecture
        if not architecture.joint_levels:
            independent_architecture = asdict(architecture)
            independent_architecture.update(
                {
                    "query_dim": 10,
                    "context_dim": 11,
                    "relative_dim": 4,
                    "output_dim": 1,
                    "joint_levels": True,
                }
            )
            self.level_models = nn.ModuleList(
                [
                    HGBSphericalResidualNeuralProcess(SRNPArchitecture(**independent_architecture))
                    for _ in range(architecture.output_dim)
                ]
            )
            return
        d_model = architecture.d_model
        self.query_encoder = nn.Sequential(
            nn.Linear(architecture.query_dim, d_model), nn.GELU(), nn.LayerNorm(d_model)
        )
        self.context_encoder = nn.Sequential(
            nn.Linear(architecture.context_dim + architecture.relative_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=architecture.num_heads,
            dim_feedforward=d_model * 2,
            dropout=architecture.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_transformer = nn.TransformerEncoder(layer, num_layers=architecture.context_layers)
        self.cross_attention = nn.ModuleList(
            [CrossAttentionBlock(d_model, architecture.num_heads, architecture.dropout) for _ in range(architecture.cross_layers)]
        )
        self.innovation_head = nn.Linear(d_model, architecture.output_dim)
        self.log_std_head = nn.Linear(d_model, architecture.output_dim)
        self.gate_head = nn.Sequential(nn.Linear(d_model + 1, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1))
        self.distance_slope_raw = nn.Parameter(torch.tensor(0.5))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.architecture.joint_levels:
            outputs = []
            for level, model in enumerate(self.level_models):
                level_batch = dict(batch)
                level_batch["query"] = torch.cat(
                    [
                        batch["query"][:, level : level + 1],
                        batch["query"][:, 3 + level : 4 + level],
                        batch["query"][:, 6 + level : 7 + level],
                        batch["query"][:, 9:16],
                    ],
                    dim=-1,
                )
                level_batch["prior"] = batch["prior"][:, level : level + 1]
                level_batch["context"] = torch.cat(
                    [
                        batch["context"][:, :, level : level + 1],
                        batch["context"][:, :, 3 + level : 4 + level],
                        batch["context"][:, :, 6 + level : 7 + level],
                        batch["context"][:, :, 9:16],
                        batch["context"][:, :, 16 + level : 17 + level],
                    ],
                    dim=-1,
                )
                level_batch["relative"] = torch.cat(
                    [batch["relative"][:, :, :3], batch["relative"][:, :, 3 + level : 4 + level]], dim=-1
                )
                outputs.append(model(level_batch))
            return {
                "residual": torch.cat([value["residual"] for value in outputs], dim=-1),
                "innovation": torch.cat([value["innovation"] for value in outputs], dim=-1),
                "std": torch.cat([value["std"] for value in outputs], dim=-1),
                "log_std": torch.cat([value["log_std"] for value in outputs], dim=-1),
                "gate": torch.stack([value["gate"] for value in outputs], dim=-1).mean(dim=-1),
            }
        query = self.query_encoder(batch["query"]).unsqueeze(1)
        context_input = torch.cat([batch["context"], batch["relative"]], dim=-1)
        context = self.context_encoder(context_input)
        valid = batch["context_mask"].bool()
        has_context = valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[~has_context, 0] = True
        context = context.masked_fill(~safe_valid.unsqueeze(-1), 0.0)
        context = self.context_transformer(context, src_key_padding_mask=~safe_valid)
        context = context.masked_fill(~safe_valid.unsqueeze(-1), 0.0)
        for block in self.cross_attention:
            query = block(query, context, key_padding_mask=~safe_valid)
        latent = query[:, 0]
        innovation = self.innovation_head(latent)
        log_std = self.log_std_head(latent).clamp(min=-5.0, max=3.0)
        count_fraction = (batch["context_count"] / float(self.architecture.nearest_context)).clamp(0.0, 1.0)
        gate_features = torch.cat([latent, count_fraction.unsqueeze(-1)], dim=-1)
        gate_logit = self.gate_head(gate_features).squeeze(-1)
        if self.architecture.use_distance_gate:
            penalty = F.softplus(self.distance_slope_raw) * torch.log1p(batch["min_distance_km"] / 250.0)
            gate_logit = gate_logit - penalty
        gate = torch.sigmoid(gate_logit) * count_fraction
        gate = torch.where(has_context, gate, torch.zeros_like(gate))
        if not self.architecture.use_context:
            gate = torch.ones_like(gate)
        prior = batch["prior"] if self.architecture.use_hgb_prior else torch.zeros_like(batch["prior"])
        residual = prior + gate.unsqueeze(-1) * innovation
        distance_term = torch.log1p(batch["min_distance_km"].clamp(max=5000.0) / 1000.0)
        sparsity_factor = 1.0 + 0.5 * (1.0 - gate) + 0.25 * distance_term + 0.25 * (1.0 - count_fraction)
        std = torch.exp(log_std) * sparsity_factor.unsqueeze(-1)
        return {
            "residual": residual,
            "innovation": innovation,
            "std": std,
            "log_std": torch.log(std.clamp_min(1e-5)),
            "gate": gate,
        }


def srnp_loss(
    output: dict[str, torch.Tensor],
    target: torch.Tensor,
    huber_weight: float = 0.2,
    use_uncertainty: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    error = target - output["residual"]
    if use_uncertainty:
        variance = output["std"].square().clamp_min(1e-6)
        nll = 0.5 * (torch.log(variance) + error.square() / variance).mean()
    else:
        nll = error.square().mean()
    huber = F.smooth_l1_loss(output["residual"], target)
    loss = nll + float(huber_weight) * huber
    return loss, {"loss": float(loss.detach()), "nll": float(nll.detach()), "huber": float(huber.detach())}


class NeuralContextResidualModel:
    def __init__(
        self,
        network: HGBSphericalResidualNeuralProcess,
        prior_model: Any,
        scalers: MappingScalers,
        calibration: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ):
        self.network = network
        self.prior_model = prior_model
        self.scalers = scalers
        self.layers = scalers.layers
        self.calibration = np.asarray(calibration or [1.0] * len(self.layers), dtype=np.float32)
        self.metadata = metadata or {}
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.network.to(self.device).eval()

    def predict_residual(self, frame: pd.DataFrame, context_frame: pd.DataFrame | None = None) -> np.ndarray:
        return self.predict(frame, context_frame=context_frame).mean

    def predict(
        self,
        target_frame: pd.DataFrame,
        context_frame: pd.DataFrame | None = None,
    ) -> MappingPrediction:
        target = _ensure_mapping_columns(target_frame)
        prior = self.prior_model.predict_residual(target)
        target_profiles = build_profile_table(target, prior_residual=prior, layers=self.layers)
        if context_frame is None or context_frame.empty:
            contexts = target_profiles.iloc[0:0].copy()
        else:
            context = _ensure_mapping_columns(context_frame)
            if "split" in context and self.metadata.get("context_splits"):
                context = context[context["split"].isin(self.metadata["context_splits"])].copy()
            context_prior = self.prior_model.predict_residual(context)
            contexts = build_profile_table(context, prior_residual=context_prior, layers=self.layers)
        dataset = SpatialEpisodeDataset(
            target_profiles,
            contexts,
            self.scalers,
            nearest_context=self.network.architecture.nearest_context,
            candidate_context=int(self.metadata.get("candidate_context", 32)),
            training=False,
            seed=int(self.metadata.get("seed", 42)),
            context_fraction=float(self.metadata.get("context_fraction", 1.0)),
            use_context=self.network.architecture.use_context,
            use_geometry=self.network.architecture.use_geometry,
            use_hgb_prior=self.network.architecture.use_hgb_prior,
        )
        loader = DataLoader(
            dataset,
            batch_size=int(self.metadata.get("inference_batch_size", 2048)),
            shuffle=False,
            num_workers=0,
        )
        profile_mean: list[np.ndarray] = []
        profile_std: list[np.ndarray] = []
        profile_gate: list[np.ndarray] = []
        profile_distance: list[np.ndarray] = []
        profile_count: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                device_batch = {key: value.to(self.device) for key, value in batch.items() if isinstance(value, torch.Tensor)}
                output = self.network(device_batch)
                profile_mean.append(output["residual"].cpu().numpy())
                profile_std.append(output["std"].cpu().numpy())
                profile_gate.append(output["gate"].cpu().numpy())
                profile_distance.append(batch["min_distance_km"].numpy())
                profile_count.append(batch["context_count"].numpy())
        if not profile_mean:
            raise ValueError("No complete target profiles were available for neural mapping prediction.")
        mean = self.scalers.denormalize_residual(np.concatenate(profile_mean)).astype(np.float32)
        std = (
            self.scalers.denormalize_residual(np.concatenate(profile_std)) * self.calibration[None, :]
        ).astype(np.float32)
        gate = np.concatenate(profile_gate).astype(np.float32)
        distance = np.concatenate(profile_distance).astype(np.float32)
        count = np.concatenate(profile_count).astype(np.float32)
        return _profiles_to_row_prediction(target_profiles, len(target), self.layers, mean, std, gate, distance, count)

    def save(self, path: str | Path, resolved_config: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        prior_dir = path / "prior"
        prior_dir.mkdir(exist_ok=True)
        self.prior_model.save(prior_dir / "model.joblib")
        torch.save(self.network.state_dict(), path / "model.pt")
        joblib.dump(self.scalers.to_dict(), path / "scalers.joblib")
        (path / "calibration.json").write_text(
            json.dumps({"scale": self.calibration.tolist()}, indent=2), encoding="utf-8"
        )
        metadata = {
            **self.metadata,
            "model_type": "hgb_srnp",
            "architecture": asdict(self.network.architecture),
            "layers": self.layers,
        }
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        if resolved_config is not None:
            import yaml

            (path / "resolved_config.yaml").write_text(
                yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True), encoding="utf-8"
            )

    @classmethod
    def load(cls, path: str | Path, device: str | torch.device | None = None) -> "NeuralContextResidualModel":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
        architecture = SRNPArchitecture(**metadata["architecture"])
        network = HGBSphericalResidualNeuralProcess(architecture)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        network.load_state_dict(state)
        scalers = MappingScalers.from_dict(joblib.load(path / "scalers.joblib"))
        calibration_path = path / "calibration.json"
        calibration = [1.0] * len(scalers.layers)
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["scale"]
        prior_model = ResidualMappingModel.load(path / "prior" / "model.joblib")
        return cls(network, prior_model, scalers, calibration=calibration, metadata=metadata, device=device)


def calibrate_uncertainty(
    residual_true: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    coverage: float = 0.9,
) -> np.ndarray:
    z = 1.6448536269514722 if np.isclose(coverage, 0.9) else 1.0
    ratio = np.abs(residual_true - residual_mean) / np.maximum(z * residual_std, 1e-6)
    scale = np.nanquantile(ratio, coverage, axis=0)
    return np.clip(scale, 0.25, 4.0).astype(np.float32)


def _ensure_mapping_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.reset_index(drop=True).copy()
    if "era5_height_km" not in out and "era5_height_m" in out:
        out["era5_height_km"] = pd.to_numeric(out["era5_height_m"], errors="coerce") / 1000.0
    if any(col not in out for col in ["lat_norm", "lon_sin", "doy_sin", "layer_norm"]):
        out = add_mapping_features(out)
    return out


def _profiles_to_row_prediction(
    profiles: pd.DataFrame,
    row_count: int,
    layers: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    gate: np.ndarray,
    distance: np.ndarray,
    count: np.ndarray,
) -> MappingPrediction:
    row_mean = np.full(row_count, np.nan, dtype=np.float32)
    row_std = np.full(row_count, np.nan, dtype=np.float32)
    row_gate = np.full(row_count, np.nan, dtype=np.float32)
    row_distance = np.full(row_count, np.nan, dtype=np.float32)
    row_count_values = np.full(row_count, np.nan, dtype=np.float32)
    for layer_index, layer in enumerate(layers):
        indices = profiles[f"_row_index_{layer}"].to_numpy(dtype=int)
        row_mean[indices] = mean[:, layer_index]
        row_std[indices] = std[:, layer_index]
        row_gate[indices] = gate
        row_distance[indices] = distance
        row_count_values[indices] = count
    if np.isnan(row_mean).any():
        raise ValueError("Neural mapping requires complete profiles for every target row.")
    return MappingPrediction(
        mean=row_mean,
        std=row_std,
        gate=row_gate,
        nearest_context_distance_km=row_distance,
        context_station_count=row_count_values,
    )
