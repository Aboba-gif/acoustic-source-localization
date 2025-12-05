from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class RoomConfig:
    Lx: float
    Ly: float
    Lz: float
    z_meas: float
    measurement_height: float | None = None


@dataclass
class GridConfig:
    nx: int
    ny: int


@dataclass
class SourcesConfig:
    n_min: int
    n_max: int
    margin_xy: float
    z_min: float
    z_max: float
    types: List[str]
    spl_ranges: Dict[str, List[float]]
    source_map_sigma_m: float = 0.2


@dataclass
class AcousticsConfig:
    c: float | None
    freqs: List[float]
    alpha: float
    floor_absorption: float
    snr_db: float | None = None

    temperature: float = 20.0
    pressure_atm: float = 101325.0
    humidity: float = 50.0

    # микрофон + АЦП
    mic_sensitivity: float = 50e-3
    mic_self_noise: float = 15.0
    mic_max_spl: float = 140.0
    adc_bits: int = 24
    adc_voltage_range: float = 10.0

    # новые параметры поглощения
    ceiling_absorption: float = 0.2
    wall_absorption_x: float = 0.2  # стены x=0 и x=Lx
    wall_absorption_y: float = 0.2  # стены y=0 и y=Ly

    # частотно-зависимое поглощение для бетона (если задано)
    # формат: {freq_hz: absorption_value}
    concrete_absorption_freq: Optional[Dict[float, float]] = None


@dataclass
class DatasetConfig:
    n_train: int
    n_val: int
    n_test: int
    h5_out_dir: str
    json_out_dir: str
    source_map_sigma_m: float = 0.2


@dataclass
class FullSimConfig:
    room: RoomConfig
    grid: GridConfig
    sources: SourcesConfig
    acoustics: AcousticsConfig
    dataset: DatasetConfig