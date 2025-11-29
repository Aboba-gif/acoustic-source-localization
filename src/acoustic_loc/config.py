from dataclasses import dataclass
from typing import List, Dict


@dataclass
class RoomConfig:
    Lx: float
    Ly: float
    Lz: float
    z_meas: float


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


@dataclass
class AcousticsConfig:
    c: float
    freqs: List[float]
    alpha: float
    floor_absorption: float
    snr_db: float


@dataclass
class DatasetConfig:
    n_train: int
    n_val: int
    n_test: int
    h5_out_dir: str
    json_out_dir: str
    source_map_sigma_m: float


@dataclass
class FullSimConfig:
    room: RoomConfig
    grid: GridConfig
    sources: SourcesConfig
    acoustics: AcousticsConfig
    dataset: DatasetConfig
