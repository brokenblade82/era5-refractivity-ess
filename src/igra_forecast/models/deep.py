from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from igra_forecast.data import ExperimentData, SplitArrays, predictions_to_frame
from igra_forecast.logging_utils import info
from igra_forecast.models.base import ForecastModel


DEEP_MODEL_NAMES = {
    "lstm",
    "gru",
    "tcn",
    "dlinear",
    "nlinear",
    "patchtst",
    "itransformer",
    "timemixer",
    "refrac_mixer",
}


class DeepForecastModel(ForecastModel):
    def fit(self, data: ExperimentData) -> None:
        common = self.params["common"]
        self.device = _select_device(common.get("device", "auto"))
        info(f"深度模型使用设备：{self.device}")
        self.model = build_torch_module(self.name, data, self.params).to(self.device)
        self.uses_priors = bool(getattr(self.model, "uses_priors", False))
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        info(f"模型参数量：总计 {total_params:,}，可训练 {trainable_params:,}")
        if self.uses_priors:
            info("当前模型启用气候态先验、Persistence 先验和目标日历特征")
        self.training_history = self._train(data, common)

    def predict(self, data: ExperimentData, split: str = "test") -> pd.DataFrame:
        split_data = data.splits[split]
        self.model.eval()
        preds: list[np.ndarray] = []
        loader = _make_loader(split_data, int(self.params["common"].get("batch_size", 64)), False, self.uses_priors)
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"预测 {split}", unit="batch"):
                batch = _move_batch(batch, self.device)
                out = self._forward_batch(batch).detach().cpu().numpy()
                preds.append(out)
        y_pred = np.concatenate(preds, axis=0)
        return predictions_to_frame(data, split, split_data.y, y_pred.astype(np.float32))

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "name": self.name,
                "params": self.params,
                "state_dict": self.model.state_dict(),
            },
            path / "model.pt",
        )

    def _train(self, data: ExperimentData, common: dict[str, Any]) -> pd.DataFrame:
        epochs = int(common.get("epochs", 30))
        batch_size = int(common.get("batch_size", 64))
        lr = float(common.get("learning_rate", 1e-3))
        weight_decay = float(common.get("weight_decay", 1e-4))
        patience = int(common.get("patience", 8))
        train_loader = _make_loader(data.train, batch_size, True, self.uses_priors)
        val_loader = _make_loader(data.val, batch_size, False, self.uses_priors)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        best_state = None
        best_val = float("inf")
        wait = 0
        rows: list[dict[str, float | int]] = []
        horizon_weights = common.get("horizon_weights")
        horizon_weights_tensor = None
        if horizon_weights:
            horizon_weights_tensor = torch.tensor(horizon_weights, dtype=torch.float32, device=self.device).view(1, -1, 1)
            info(f"启用 horizon-weighted loss：{horizon_weights}")

        info(
            f"训练参数：epochs={epochs}, batch_size={batch_size}, "
            f"lr={lr}, weight_decay={weight_decay}, patience={patience}"
        )
        info(f"训练 batch 数：{len(train_loader)}，验证 batch 数：{len(val_loader)}")

        epoch_bar = tqdm(range(1, epochs + 1), desc=f"训练 {self.name}", unit="epoch")
        for epoch in epoch_bar:
            self.model.train()
            train_losses = []
            batch_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
            for batch in batch_bar:
                batch = _move_batch(batch, self.device)
                yb = batch[1]
                optimizer.zero_grad(set_to_none=True)
                pred = self._forward_batch(batch)
                loss = _forecast_loss(pred, yb, horizon_weights_tensor)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_value = float(loss.detach().cpu())
                train_losses.append(loss_value)
                batch_bar.set_postfix(loss=f"{loss_value:.5f}")

            val_loss = self._evaluate_loss(val_loader, horizon_weights_tensor)
            train_loss = float(np.mean(train_losses))
            rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

            if val_loss < best_val:
                best_val = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
                info(f"Epoch {epoch}: 验证损失改进为 {best_val:.6f}，已记录最佳模型")
            else:
                wait += 1
                info(f"Epoch {epoch}: 验证损失 {val_loss:.6f} 未改进，early stopping 计数 {wait}/{patience}")
                if wait >= patience:
                    info(f"触发 early stopping：连续 {patience} 轮验证损失未改进")
                    break
            epoch_bar.set_postfix(train=f"{train_loss:.5f}", val=f"{val_loss:.5f}", best=f"{best_val:.5f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)
            info(f"已恢复验证损失最优模型，best_val={best_val:.6f}")
        return pd.DataFrame(rows)

    def _evaluate_loss(self, loader: DataLoader, horizon_weights: torch.Tensor | None) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in loader:
                batch = _move_batch(batch, self.device)
                yb = batch[1]
                losses.append(float(_forecast_loss(self._forward_batch(batch), yb, horizon_weights).detach().cpu()))
        return float(np.mean(losses))

    def _forward_batch(self, batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
        if self.uses_priors:
            xb, _, climatology_prior, persistence_prior, calendar_features = batch
            return self.model(xb, climatology_prior, persistence_prior, calendar_features)
        xb = batch[0]
        return self.model(xb)


def build_torch_module(name: str, data: ExperimentData, params: dict[str, Any]) -> nn.Module:
    n_features = len(data.input_cols)
    n_targets = len(data.target_cols)
    input_length = data.input_length
    horizon_count = len(data.horizons)
    target_indices = [data.input_cols.index(col) for col in data.target_cols]

    if name == "lstm":
        return RecurrentForecaster("lstm", n_features, n_targets, horizon_count, params[name])
    if name == "gru":
        return RecurrentForecaster("gru", n_features, n_targets, horizon_count, params[name])
    if name == "tcn":
        return TcnForecaster(n_features, n_targets, horizon_count, params[name])
    if name == "dlinear":
        return DLinearForecaster(input_length, n_features, n_targets, horizon_count, params[name])
    if name == "nlinear":
        return NLinearForecaster(input_length, n_features, n_targets, horizon_count, target_indices, params[name])
    if name == "patchtst":
        return PatchTstLite(input_length, n_features, n_targets, horizon_count, params[name])
    if name == "itransformer":
        return ITransformerLite(input_length, n_features, n_targets, horizon_count, params[name])
    if name == "timemixer":
        return TimeMixerLite(input_length, n_features, n_targets, horizon_count, params[name])
    if name == "refrac_mixer":
        return RefracMixer(input_length, n_features, n_targets, horizon_count, params[name])
    raise ValueError(f"Unsupported deep model: {name}")


class RecurrentForecaster(nn.Module):
    def __init__(self, cell: str, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 64))
        layers = int(cfg.get("num_layers", 2))
        dropout = float(cfg.get("dropout", 0.1)) if layers > 1 else 0.0
        rnn_cls = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn_cls(n_features, hidden, num_layers=layers, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, horizons * n_targets))
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        y = self.head(out[:, -1])
        return y.view(x.size(0), self.horizons, self.n_targets)


class TcnForecaster(nn.Module):
    def __init__(self, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 64))
        layers = int(cfg.get("num_layers", 3))
        kernel = int(cfg.get("kernel_size", 3))
        dropout = float(cfg.get("dropout", 0.1))
        blocks = []
        in_ch = n_features
        for idx in range(layers):
            dilation = 2**idx
            blocks.append(CausalConvBlock(in_ch, hidden, kernel, dilation, dropout))
            in_ch = hidden
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, horizons * n_targets)
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x.transpose(1, 2)).transpose(1, 2)
        y = self.head(z[:, -1])
        return y.view(x.size(0), self.horizons, self.n_targets)


class CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.chomp = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        if self.chomp > 0:
            y = y[:, :, : -self.chomp]
        y = self.drop(self.act(self.norm(y)))
        return y + self.proj(x)


class DLinearForecaster(nn.Module):
    def __init__(self, input_length: int, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.moving_avg = int(cfg.get("moving_avg", 7))
        hidden = int(cfg.get("hidden_size", 128))
        dropout = float(cfg.get("dropout", 0.0))
        in_dim = input_length * n_features * 2
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizons * n_targets),
        )
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = moving_average(x, self.moving_avg)
        seasonal = x - trend
        z = torch.cat([trend, seasonal], dim=-1).flatten(1)
        y = self.head(z)
        return y.view(x.size(0), self.horizons, self.n_targets)


class NLinearForecaster(nn.Module):
    def __init__(
        self,
        input_length: int,
        n_features: int,
        n_targets: int,
        horizons: int,
        target_indices: list[int],
        cfg: dict[str, Any],
    ) -> None:
        super().__init__()
        hidden = int(cfg.get("hidden_size", 128))
        dropout = float(cfg.get("dropout", 0.0))
        self.target_indices = target_indices
        self.head = nn.Sequential(
            nn.Linear(input_length * n_features, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizons * n_targets),
        )
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1:, self.target_indices]
        centered = x - x[:, -1:, :]
        delta = self.head(centered.flatten(1)).view(x.size(0), self.horizons, self.n_targets)
        return delta + last


class PatchTstLite(nn.Module):
    def __init__(self, input_length: int, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        patch_len = int(cfg.get("patch_len", 8))
        stride = int(cfg.get("stride", 4))
        d_model = int(cfg.get("d_model", 64))
        heads = int(cfg.get("num_heads", 4))
        layers = int(cfg.get("num_layers", 2))
        ff = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))
        self.patch_len = patch_len
        self.stride = stride
        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        patch_count = 1 + max(0, (input_length - patch_len) // stride)
        self.pos = nn.Parameter(torch.zeros(1, patch_count, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, horizons * n_targets))
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).flatten(2)
        z = self.patch_proj(patches) + self.pos[:, : patches.size(1)]
        z = self.encoder(z).mean(dim=1)
        y = self.head(z)
        return y.view(x.size(0), self.horizons, self.n_targets)


class ITransformerLite(nn.Module):
    def __init__(self, input_length: int, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        d_model = int(cfg.get("d_model", 64))
        heads = int(cfg.get("num_heads", 4))
        layers = int(cfg.get("num_layers", 2))
        ff = int(cfg.get("dim_feedforward", 128))
        dropout = float(cfg.get("dropout", 0.1))
        self.value_proj = nn.Linear(input_length, d_model)
        self.var_embed = nn.Parameter(torch.zeros(1, n_features, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, heads, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        self.head = nn.Sequential(nn.LayerNorm(n_features * d_model), nn.Linear(n_features * d_model, horizons * n_targets))
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.value_proj(x.transpose(1, 2)) + self.var_embed
        z = self.encoder(tokens).flatten(1)
        y = self.head(z)
        return y.view(x.size(0), self.horizons, self.n_targets)


class TimeMixerLite(nn.Module):
    def __init__(self, input_length: int, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        scales = [int(s) for s in cfg.get("scales", [1, 3, 7])]
        hidden = int(cfg.get("hidden_size", 128))
        dropout = float(cfg.get("dropout", 0.1))
        self.scales = scales
        self.head = nn.Sequential(
            nn.Linear(input_length * n_features * len(scales), hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, horizons * n_targets),
        )
        self.horizons = horizons
        self.n_targets = n_targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = []
        for scale in self.scales:
            if scale <= 1:
                branches.append(x)
            else:
                branches.append(moving_average(x, scale))
        y = self.head(torch.cat(branches, dim=-1).flatten(1))
        return y.view(x.size(0), self.horizons, self.n_targets)


class RefracMixer(nn.Module):
    uses_priors = True

    def __init__(self, input_length: int, n_features: int, n_targets: int, horizons: int, cfg: dict[str, Any]) -> None:
        super().__init__()
        self.input_length = input_length
        self.n_features = n_features
        self.n_targets = n_targets
        self.horizons = horizons
        self.scales = [int(s) for s in cfg.get("scales", [1, 3, 7, 15, 30])]
        self.use_multiscale = bool(cfg.get("use_multiscale", True))
        self.use_series_core = bool(cfg.get("use_series_core", True))
        self.use_target_variable_gating = bool(cfg.get("use_target_variable_gating", False))
        self.use_climatology_prior = bool(cfg.get("use_climatology_prior", True))
        self.use_persistence_prior = bool(cfg.get("use_persistence_prior", True))
        self.gate_temperature = float(cfg.get("gate_temperature", 1.0))
        self.residual_scale = float(cfg.get("residual_scale", 1.0))
        self.use_learnable_target_residual_scale = bool(cfg.get("use_learnable_target_residual_scale", False))
        d_model = int(cfg.get("d_model", 64))
        hidden = int(cfg.get("hidden_size", 128))
        dropout = float(cfg.get("dropout", 0.15))
        active_scales = self.scales if self.use_multiscale else [1]
        self.active_scales = active_scales
        token_in = input_length * len(active_scales) * 2
        self.variable_encoder = nn.Sequential(
            nn.Linear(token_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model),
        )
        self.series_core = SeriesCoreFusion(d_model, hidden, dropout) if self.use_series_core else nn.Identity()
        self.target_variable_gate = (
            TargetAwareVariableGating(n_features, n_targets, d_model, hidden, dropout)
            if self.use_target_variable_gating
            else None
        )
        self.calendar_encoder = nn.Sequential(nn.Linear(2, d_model), nn.GELU(), nn.LayerNorm(d_model))
        context_dim = (n_targets if self.use_target_variable_gating else n_features) * d_model + d_model
        self.residual_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(context_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_targets),
                )
                for _ in range(horizons)
            ]
        )
        self.gate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(context_dim, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, n_targets * 3),
                )
                for _ in range(horizons)
            ]
        )
        self.target_residual_log_scale = nn.Parameter(torch.zeros(n_targets))
        gate_bias = cfg.get("gate_bias")
        if gate_bias is None:
            gate_bias = [[0.0, 0.0, 0.0] for _ in range(horizons)]
        gate_bias_tensor = torch.tensor(gate_bias, dtype=torch.float32)
        if gate_bias_tensor.shape != (horizons, 3):
            raise ValueError(f"refrac_mixer.gate_bias must have shape [{horizons}, 3].")
        self.register_buffer("gate_bias", gate_bias_tensor)

    def forward(
        self,
        x: torch.Tensor,
        climatology_prior: torch.Tensor,
        persistence_prior: torch.Tensor,
        calendar_features: torch.Tensor,
    ) -> torch.Tensor:
        tokens = self._encode_variables(x)
        tokens = self.series_core(tokens)
        target_context = self.target_variable_gate(tokens) if self.target_variable_gate is not None else tokens
        flat_context = target_context.flatten(1)
        outputs = []
        for horizon_idx in range(self.horizons):
            cal = self.calendar_encoder(calendar_features[:, horizon_idx])
            context = torch.cat([flat_context, cal], dim=-1)
            residual = self.residual_heads[horizon_idx](context)
            persistence = persistence_prior[:, horizon_idx]
            climatology = climatology_prior[:, horizon_idx]
            if not self.use_persistence_prior:
                persistence = climatology
            if not self.use_climatology_prior:
                climatology = persistence
            target_scale = 1.0
            if self.use_learnable_target_residual_scale:
                target_scale = torch.sigmoid(self.target_residual_log_scale).view(1, -1) * 2.0
            neural_candidate = persistence + self.residual_scale * target_scale * residual
            candidates = torch.stack([climatology, persistence, neural_candidate], dim=-1)
            gates = self.gate_heads[horizon_idx](context).view(x.size(0), self.n_targets, 3)
            gates = gates + self.gate_bias[horizon_idx].view(1, 1, 3)
            weights = torch.softmax(gates / self.gate_temperature, dim=-1)
            outputs.append((candidates * weights).sum(dim=-1))
        return torch.stack(outputs, dim=1)

    def _encode_variables(self, x: torch.Tensor) -> torch.Tensor:
        per_scale = []
        for scale in self.active_scales:
            smooth = moving_average(x, scale)
            residual = x - smooth
            per_scale.extend([smooth, residual])
        z = torch.cat(per_scale, dim=1)
        z = z.transpose(1, 2).flatten(2)
        return self.variable_encoder(z)


class SeriesCoreFusion(nn.Module):
    def __init__(self, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.core = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, d_model))
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        core = self.core(tokens.mean(dim=1, keepdim=True)).expand_as(tokens)
        return self.fuse(torch.cat([tokens, core], dim=-1)) + tokens


class TargetAwareVariableGating(nn.Module):
    def __init__(self, n_features: int, n_targets: int, d_model: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.n_targets = n_targets
        self.query = nn.Parameter(torch.randn(n_targets, d_model) * 0.02)
        self.score = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.post = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU())

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch, n_features, d_model = tokens.shape
        query = self.query.view(1, self.n_targets, 1, d_model).expand(batch, -1, n_features, -1)
        expanded_tokens = tokens.view(batch, 1, n_features, d_model).expand(-1, self.n_targets, -1, -1)
        scores = self.score(torch.cat([expanded_tokens, query], dim=-1)).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("btf,bfd->btd", weights, tokens)
        return self.post(mixed)


def moving_average(x: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    pad_left = (kernel - 1) // 2
    pad_right = kernel - 1 - pad_left
    z = x.transpose(1, 2)
    z = torch.nn.functional.pad(z, (pad_left, pad_right), mode="replicate")
    z = torch.nn.functional.avg_pool1d(z, kernel_size=kernel, stride=1)
    return z.transpose(1, 2)


def _forecast_loss(pred: torch.Tensor, truth: torch.Tensor, horizon_weights: torch.Tensor | None) -> torch.Tensor:
    loss = (pred - truth) ** 2
    if horizon_weights is not None:
        loss = loss * horizon_weights
    return loss.mean()


def _make_loader(split: SplitArrays, batch_size: int, shuffle: bool, include_priors: bool) -> DataLoader:
    tensors: list[torch.Tensor] = [torch.from_numpy(split.x).float(), torch.from_numpy(split.y).float()]
    if include_priors:
        tensors.extend(
            [
                torch.from_numpy(split.climatology_prior).float(),
                torch.from_numpy(split.persistence_prior).float(),
                torch.from_numpy(split.calendar_features).float(),
            ]
        )
    dataset = TensorDataset(*tensors)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(tensor.to(device) for tensor in batch)


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
