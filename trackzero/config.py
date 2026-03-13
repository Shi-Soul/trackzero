"""Configuration dataclasses with YAML loading."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class PendulumConfig:
    link_length: float = 0.5
    link_mass: float = 1.0
    link_inertia: list[float] = field(default_factory=lambda: [0.02, 0.02, 0.001])
    joint_damping: float = 0.5
    tau_max: float = 5.0
    gravity: float = 9.81


@dataclass
class SimulationConfig:
    dt: float = 0.002
    control_dt: float = 0.01
    integrator: str = "RK4"

    @property
    def substeps(self) -> int:
        return round(self.control_dt / self.dt)


@dataclass
class MultisineConfig:
    k_range: list[int] = field(default_factory=lambda: [3, 8])
    freq_range: list[float] = field(default_factory=lambda: [0.1, 3.0])


@dataclass
class InitialStateConfig:
    q_range: list[float] = field(default_factory=lambda: [-3.14159265, 3.14159265])
    v_range: list[float] = field(default_factory=lambda: [-3.0, 3.0])


@dataclass
class DatasetConfig:
    n_train: int = 80000
    n_test: int = 20000
    trajectory_duration: float = 5.0
    multisine: MultisineConfig = field(default_factory=MultisineConfig)
    initial_state: InitialStateConfig = field(default_factory=InitialStateConfig)
    num_workers: int = 8
    chunk_size: int = 100


@dataclass
class EvalConfig:
    mse_velocity_weight: float = 0.1


@dataclass
class Config:
    pendulum: PendulumConfig = field(default_factory=PendulumConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _dataclass_from_dict(cls, d: dict):
    """Recursively build a dataclass from a nested dict."""
    if not dataclasses.is_dataclass(cls):
        return d
    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs = {}
    for k, v in d.items():
        if k not in field_types:
            continue
        ft = field_types[k]
        # Resolve string annotations
        if isinstance(ft, str):
            ft = eval(ft, {**globals(), **{c.__name__: c for c in [
                PendulumConfig, SimulationConfig, MultisineConfig,
                InitialStateConfig, DatasetConfig, EvalConfig, Config,
            ]}})
        if dataclasses.is_dataclass(ft) and isinstance(v, dict):
            kwargs[k] = _dataclass_from_dict(ft, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_config(path: Optional[str | Path] = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        return Config()
    with open(path) as f:
        raw = yaml.safe_load(f)
    return _dataclass_from_dict(Config, raw)


def save_config(cfg: Config, path: str | Path) -> None:
    """Save config to YAML file."""
    with open(path, "w") as f:
        yaml.dump(dataclasses.asdict(cfg), f, default_flow_style=False, sort_keys=False)
