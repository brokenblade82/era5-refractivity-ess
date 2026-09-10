from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

from tqdm.auto import tqdm


def configure_console_encoding() -> None:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTHONUTF8", "1")
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def info(message: str) -> None:
    tqdm.write(f"[IGRA] {message}")


def section(title: str) -> None:
    tqdm.write("")
    tqdm.write(f"[IGRA] ===== {title} =====")


def format_path(path: str | Path) -> str:
    return str(Path(path))


def print_config_summary(cfg: dict[str, Any]) -> None:
    section("实验配置")
    info(f"模型名称：{cfg['model']['name']}")
    info(f"数据文件：{cfg['data']['path']}")
    info(f"数据场景：{cfg['data'].get('scenario', 'full')}")
    info(f"输入窗口长度：{cfg['task']['input_length']} 天")
    info(f"预测步长：{cfg['task']['horizons']}")
    info(f"预测目标：{', '.join(cfg['data']['target_cols'])}")
    info(f"输入变量数：{len(cfg['data']['input_cols'])}")
    info(f"随机种子：{cfg['project'].get('seed', 42)}")
    if cfg.get("debug"):
        active = {k: v for k, v in cfg["debug"].items() if v is not None}
        if active:
            info(f"调试样本限制：{active}")


def print_data_summary(summary: dict[str, Any]) -> None:
    section("数据摘要")
    info(f"原始记录数：{summary['rows']}")
    info(f"时间范围：{summary['start']} 至 {summary['end']}")
    info(f"缺失值数量：{summary['missing_values']}")
    for split_name, split_summary in summary["splits"].items():
        info(
            f"{split_name}: {split_summary['rows']} 行，"
            f"{split_summary['start']} 至 {split_summary['end']}"
        )


def print_window_summary(train_n: int, val_n: int, test_n: int, input_length: int, horizons: list[int]) -> None:
    section("监督样本")
    info(f"输入时间步数：{input_length}")
    info(f"输出预测步长：{horizons}")
    info(f"训练窗口数：{train_n}")
    info(f"验证窗口数：{val_n}")
    info(f"测试窗口数：{test_n}")
