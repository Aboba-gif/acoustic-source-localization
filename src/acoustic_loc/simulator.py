# src/acoustic_loc/simulator.py
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import FullSimConfig


class AcousticSimulator:

    def __init__(self, cfg: FullSimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

        # --- Геометрия и сетка ---
        self.room_length = cfg.room.Lx
        self.room_width = cfg.room.Ly
        self.room_height = cfg.room.Lz
        self.measurement_height = cfg.room.measurement_height or cfg.room.z_meas

        self.grid_x = cfg.grid.nx
        self.grid_y = cfg.grid.ny

        # шаги сетки
        self.dx = self.room_length / self.grid_x
        self.dy = self.room_width / self.grid_y

        x = np.linspace(0, self.room_length, self.grid_x, dtype=np.float32)
        y = np.linspace(0, self.room_width, self.grid_y, dtype=np.float32)
        self.X, self.Y = np.meshgrid(x, y, indexing="ij")
        self.Z = np.ones_like(self.X) * self.measurement_height

        # --- Акустика и микрофон ---
        self.config_ac = cfg.acoustics
        self.center_frequencies = list(self.config_ac.freqs)
        self.c = self._calculate_sound_speed()
        self.wall_absorption = self.config_ac.floor_absorption
        self.source_map_sigma_m = self.cfg.dataset.source_map_sigma_m

        self._precompute_params()

    # ---------------- публичный API ----------------

    def generate_scene(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Возвращает (pressure_complex_noisy, source_map, ground_truth_dict)
        """
        frequency = float(self.rng.choice(self.center_frequencies))

        # число источников: [n_min, n_max] включительно
        n_sources = int(self.rng.integers(self.cfg.sources.n_min, self.cfg.sources.n_max + 1))

        sources = [self._generate_source(frequency) for _ in range(n_sources)]

        pressure_clean = self._compute_pressure_field(sources, frequency)
        pressure_noisy = self._add_noise(pressure_clean, frequency)
        source_map = self._create_source_density_map(sources)

        ground_truth = {
            "frequency": int(frequency),
            "sound_speed": float(self.c),
            "sources": [
                {
                    "type": s["type"],
                    "position": {
                        "x": float(s["position"]["x"]),
                        "y": float(s["position"]["y"]),
                        "z": float(s["position"]["z"]),
                    },
                    "spl_db": float(s["spl_db"]),
                }
                for s in sources
            ],
            "room": asdict(self.cfg.room),
            "acoustics": {
                "temperature": self.config_ac.temperature,
                "pressure_atm": self.config_ac.pressure_atm,
                "humidity": self.config_ac.humidity,
                "wall_absorption": self.wall_absorption,
            },
        }

        return pressure_noisy, source_map, ground_truth

    def save_scene(
        self,
        pressure: np.ndarray,
        source_map: np.ndarray,
        meta: Dict[str, Any],
        json_path: Path,
    ) -> None:
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # ---------------- внутренние методы ----------------

    def _calculate_sound_speed(self) -> float:
        """
        Как в ноутбуке:
        T = temperature + 273.15
        c = 331.3 * sqrt(T / 273.15)
        """
        T = self.config_ac.temperature + 273.15
        return 331.3 * np.sqrt(T / 273.15)

    def _precompute_params(self) -> None:
        self.wave_numbers: Dict[float, float] = {}
        self.absorption_coeffs: Dict[float, float] = {}
        self.mic_frequency_response: Dict[float, float] = {}

        for f in self.center_frequencies:
            f = float(f)
            self.wave_numbers[f] = 2.0 * np.pi * f / self.c
            # alpha ~ (f/1000)^2
            self.absorption_coeffs[f] = 1e-5 * (f / 1000.0) ** 2
            # простой frequency response
            self.mic_frequency_response[f] = 1.0 + 0.05 * np.sin(np.log10(f / 1000.0))

    def _generate_source(self, frequency: float) -> Dict[str, Any]:
        """
        Перенос _generate_source из ноутбука:
        - случайный тип
        - SPL в диапазоне source_types[stype]['spl_range']
        - случайная фаза
        - спектр: amplitude / sqrt(f/1000)
        """
        # тип источника
        stype = self.rng.choice(self.cfg.sources.types)
        spl_low, spl_high = self.cfg.sources.spl_ranges[stype]
        spl_db = float(self.rng.uniform(spl_low, spl_high))

        margin = self.cfg.sources.margin_xy
        x = float(self.rng.uniform(margin, self.room_length - margin))
        y = float(self.rng.uniform(margin, self.room_width - margin))
        z = float(self.rng.uniform(self.cfg.sources.z_min, self.cfg.sources.z_max))

        p_ref = 20e-6
        amplitude = p_ref * 10.0 ** (spl_db / 20.0)
        phase = float(self.rng.uniform(0.0, 2.0 * np.pi))

        spectrum = {f: amplitude / np.sqrt(f / 1000.0) for f in self.center_frequencies}

        return {
            "type": stype,
            "position": {"x": x, "y": y, "z": z},
            "amplitude": amplitude,
            "phase": phase,
            "spl_db": spl_db,
            "spectrum": spectrum,
        }

    def _compute_pressure_field(self, sources: List[Dict[str, Any]], frequency: float) -> np.ndarray:
        """
        Поле от всех источников + отражение от пола.
        """
        pressure = np.zeros((self.grid_x, self.grid_y), dtype=np.complex128)
        frequency = float(frequency)
        k = self.wave_numbers[frequency]
        alpha = self.absorption_coeffs[frequency]

        for src in sources:
            x_src = src["position"]["x"]
            y_src = src["position"]["y"]
            z_src = src["position"]["z"]

            amp = src["spectrum"].get(frequency, src["amplitude"] * 0.1)
            Q = amp * np.exp(1j * src["phase"])

            # прямой путь
            r_direct = np.sqrt(
                (self.X - x_src) ** 2
                + (self.Y - y_src) ** 2
                + (self.Z - z_src) ** 2
            )
            r_direct = np.maximum(r_direct, 0.01)
            G_direct = (
                np.exp(1j * k * r_direct)
                / (4.0 * np.pi * r_direct)
                * np.exp(-alpha * r_direct)
            )

            # отражение от пола
            r_floor = np.sqrt(
                (self.X - x_src) ** 2
                + (self.Y - y_src) ** 2
                + (self.Z + z_src) ** 2
            )
            r_floor = np.maximum(r_floor, 0.01)
            G_floor = (
                (1.0 - self.wall_absorption)
                * np.exp(1j * k * r_floor)
                / (4.0 * np.pi * r_floor)
                * np.exp(-alpha * r_floor)
            )

            pressure += Q * (G_direct + G_floor)

        return pressure

    def _add_noise(self, pressure_field: np.ndarray, frequency: float) -> np.ndarray:
        """
        Добавляет микрофонный шум и клиппинг.
        """
        shape = pressure_field.shape
        p_ref = 20e-6

        frequency = float(frequency)
        noise_spl = self.config_ac.mic_self_noise + 10.0 * np.log10(
            frequency / 1000.0
        )
        thermal_noise_level = p_ref * 10.0 ** (noise_spl / 20.0)

        # комплексный белый шум
        thermal_noise = thermal_noise_level * (
            self.rng.standard_normal(size=shape)
            + 1j * self.rng.standard_normal(size=shape)
        ) / np.sqrt(2.0)

        mic_response = self.mic_frequency_response.get(frequency, 1.0)
        noisy_field = pressure_field * mic_response + thermal_noise

        # клиппинг по максимальному SPL микрофона
        max_pressure = p_ref * 10.0 ** (self.config_ac.mic_max_spl / 20.0)
        np.clip(noisy_field.real, -max_pressure, max_pressure, out=noisy_field.real)
        np.clip(noisy_field.imag, -max_pressure, max_pressure, out=noisy_field.imag)

        return noisy_field

    def _create_source_density_map(self, sources: List[Dict[str, Any]]) -> np.ndarray:
        """
        Как _create_source_density_map в ноутбуке:
        - сетка [grid_x, grid_y], индексирование 'ij'
        - плотность = 1 в ячейке источника, затем Gaussian с sigma в метрах/px.
        """
        density_map = np.zeros((self.grid_x, self.grid_y), dtype=np.float32)
        sigma_px = self.source_map_sigma_m / self.dx

        for src in sources:
            ix = int(src["position"]["x"] / self.dx)
            iy = int(src["position"]["y"] / self.dy)
            if 0 <= ix < self.grid_x and 0 <= iy < self.grid_y:
                density_map[ix, iy] = 1.0

        if np.any(density_map):
            density_map = gaussian_filter(density_map, sigma=sigma_px)
            density_map /= density_map.max()

        return density_map.astype(np.float32)
