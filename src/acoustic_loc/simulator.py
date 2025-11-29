import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from scipy.ndimage import gaussian_filter

from .config import FullSimConfig


class AcousticSimulator:
    def __init__(self, cfg: FullSimConfig, rng: np.random.Generator | None = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()

        self.x = np.linspace(0.0, cfg.room.Lx, cfg.grid.nx)
        self.y = np.linspace(0.0, cfg.room.Ly, cfg.grid.ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y, indexing="xy")

    # ---------- публичный API ----------

    def generate_scene(self) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Генерирует одну сцену: комплексное поле, source_map и метаданные."""
        meta = self._sample_scene_metadata()
        pressure = self._compute_pressure_field(meta)
        pressure_noisy = self._add_noise(pressure)

        source_map = self._build_source_map(meta)

        return pressure_noisy, source_map, meta

    # ---------- внутренние методы ----------

    def _sample_scene_metadata(self) -> Dict:
        cfg = self.cfg
        n_src = self.rng.integers(cfg.sources.n_min, cfg.sources.n_max + 1)

        freq = float(self.rng.choice(cfg.acoustics.freqs))
        srcs: List[Dict] = []
        for _ in range(n_src):
            stype = self.rng.choice(cfg.sources.types)
            spl_low, spl_high = cfg.sources.spl_ranges[stype]
            spl = float(self.rng.uniform(spl_low, spl_high))

            x = float(
                self.rng.uniform(
                    cfg.sources.margin_xy,
                    cfg.room.Lx - cfg.sources.margin_xy,
                )
            )
            y = float(
                self.rng.uniform(
                    cfg.sources.margin_xy,
                    cfg.room.Ly - cfg.sources.margin_xy,
                )
            )
            z = float(self.rng.uniform(cfg.sources.z_min, cfg.sources.z_max))

            srcs.append(
                {
                    "type": stype,
                    "spl": spl,
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )

        meta = {
            "freq": freq,
            "sources": srcs,
            "room": asdict(self.cfg.room),
            "acoustics": {
                "c": self.cfg.acoustics.c,
                "alpha": self.cfg.acoustics.alpha,
                "floor_absorption": self.cfg.acoustics.floor_absorption,
            },
        }
        return meta

    def _compute_pressure_field(self, meta: Dict) -> np.ndarray:
        """Фундаментальное решение Гельмгольца + отражение от пола."""
        freq = meta["freq"]
        k = 2.0 * math.pi * freq / self.cfg.acoustics.c
        alpha = self.cfg.acoustics.alpha
        R = 1.0 - self.cfg.acoustics.floor_absorption
        z_meas = self.cfg.room.z_meas

        p = np.zeros_like(self.xx, dtype=np.complex128)

        for src in meta["sources"]:
            xs, ys, zs = src["x"], src["y"], src["z"]
            Q = self._spl_to_amplitude(src["spl"], ref_dist=1.0)  # адаптируй под свой код

            r_dir = np.sqrt((self.xx - xs) ** 2 + (self.yy - ys) ** 2 + (z_meas - zs) ** 2)
            r_refl = np.sqrt(
                (self.xx - xs) ** 2 + (self.yy - ys) ** 2 + (z_meas + zs) ** 2
            )

            # избегаем деления на ноль в точной позиции источника
            r_dir = np.maximum(r_dir, 1e-4)
            r_refl = np.maximum(r_refl, 1e-4)

            term_dir = np.exp(1j * k * r_dir) * np.exp(-alpha * r_dir) / (4.0 * math.pi * r_dir)
            term_refl = (
                R
                * np.exp(1j * k * r_refl)
                * np.exp(-alpha * r_refl)
                / (4.0 * math.pi * r_refl)
            )
            p += Q * (term_dir + term_refl)

        return p

    def _spl_to_amplitude(self, spl_db: float, ref_dist: float = 1.0) -> float:
        """Перевод SPL в амплитуду Q. Здесь оставлена простая заглушка —
        вставь свою формулу из ноутбука."""
        ref_p = 2e-5  # Па
        p_rms = ref_p * 10 ** (spl_db / 20.0)
        # дальше можешь связать это с Q через своё нормирование.
        return p_rms

    def _add_noise(self, p: np.ndarray) -> np.ndarray:
        snr_db = self.cfg.acoustics.snr_db
        if snr_db is None:
            return p

        signal_power = np.mean(np.abs(p) ** 2)
        snr_lin = 10 ** (snr_db / 10.0)
        noise_power = signal_power / snr_lin
        sigma = math.sqrt(noise_power / 2.0)  # комплексный шум

        noise = sigma * (
            np.random.normal(size=p.shape) + 1j * np.random.normal(size=p.shape)
        )
        return p + noise

    def _build_source_map(self, meta: Dict) -> np.ndarray:
        cfg = self.cfg
        sigma_x_pix = cfg.dataset.source_map_sigma_m / (self.x[1] - self.x[0])
        sigma_y_pix = cfg.dataset.source_map_sigma_m / (self.y[1] - self.y[0])

        src_map = np.zeros_like(self.xx, dtype=np.float32)

        for src in meta["sources"]:
            j = int(round(src["x"] / (self.x[1] - self.x[0])))
            i = int(round(src["y"] / (self.y[1] - self.y[0])))
            i = np.clip(i, 0, cfg.grid.ny - 1)
            j = np.clip(j, 0, cfg.grid.nx - 1)
            src_map[i, j] = 1.0

        src_map = gaussian_filter(src_map, sigma=(sigma_y_pix, sigma_x_pix))
        if src_map.max() > 0:
            src_map /= src_map.max()
        return src_map.astype(np.float32)

    def save_scene(
        self,
        pressure: np.ndarray,
        source_map: np.ndarray,
        meta: Dict,
        json_path: Path,
    ) -> None:
        json_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
