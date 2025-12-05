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

        # базовая высота + небольшой jitter
        base_z_meas = cfg.room.measurement_height or cfg.room.z_meas
        jitter_z_meas = 0.02  # ±2 см
        self.measurement_height = base_z_meas + float(
            self.rng.uniform(-jitter_z_meas, jitter_z_meas)
        )

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

        # скалярные значения (если нет частотно-зависимой таблички)
        self.floor_absorption = self.config_ac.floor_absorption
        self.ceiling_absorption = self.config_ac.ceiling_absorption
        self.wall_absorption_x = self.config_ac.wall_absorption_x
        self.wall_absorption_y = self.config_ac.wall_absorption_y

        self.source_map_sigma_m = self.cfg.dataset.source_map_sigma_m
        self.max_image_order = 2  # можно вынести в конфиг

        self._precompute_params()

    # ---------------- публичный API ----------------

    def generate_scene(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Возвращает (pressure_complex_noisy, source_map, ground_truth_dict)
        """
        frequency = float(self.rng.choice(self.center_frequencies))

        # число источников: [n_min, n_max] включительно
        n_sources = int(
            self.rng.integers(self.cfg.sources.n_min, self.cfg.sources.n_max + 1)
        )
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
                "floor_absorption": self.floor_absorption,
                "ceiling_absorption": self.ceiling_absorption,
                "wall_absorption_x": self.wall_absorption_x,
                "wall_absorption_y": self.wall_absorption_y,
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
        T = temperature + 273.15
        c = 331.3 * sqrt(T / 273.15)
        """
        T = self.config_ac.temperature + 273.15
        return 331.3 * np.sqrt(T / 273.15)

    def _precompute_params(self) -> None:
        self.wave_numbers: Dict[float, float] = {}
        self.absorption_coeffs: Dict[float, float] = {}
        self.mic_frequency_response: Dict[float, float] = {}

        concrete_abs_map = self.config_ac.concrete_absorption_freq or {}

        for f in self.center_frequencies:
            f = float(f)
            self.wave_numbers[f] = 2.0 * np.pi * f / self.c

            # alpha (затухание в воздухе) пока оставляем твоё приближение
            self.absorption_coeffs[f] = 1e-5 * (f / 1000.0) ** 2

            # частотно-зависимое поглощение бетона
            if concrete_abs_map:
                # если частота есть в словаре – берём её
                alpha_surface = concrete_abs_map.get(f, list(concrete_abs_map.values())[0])
                # используем одно и то же для всех поверхностей
                self.floor_absorption_f = self.floor_absorption_f if hasattr(self, "floor_absorption_f") else {}
                self.ceiling_absorption_f = self.ceiling_absorption_f if hasattr(self, "ceiling_absorption_f") else {}
                self.wall_absorption_x_f = self.wall_absorption_x_f if hasattr(self, "wall_absorption_x_f") else {}
                self.wall_absorption_y_f = self.wall_absorption_y_f if hasattr(self, "wall_absorption_y_f") else {}

                # вместо прямого присваивания alpha_surface:
                jitter_abs = 0.01
                alpha_j = np.clip(
                    alpha_surface + self.rng.uniform(-jitter_abs, jitter_abs),
                    0.0,
                    1.0,
                )

                self.floor_absorption_f[f] = alpha_j
                self.ceiling_absorption_f[f] = alpha_j
                self.wall_absorption_x_f[f] = alpha_j
                self.wall_absorption_y_f[f] = alpha_j

            else:
                # если словарь не задан — просто используем скалярные значения из конфига
                self.floor_absorption_f = getattr(self, "floor_absorption_f", {})
                self.ceiling_absorption_f = getattr(self, "ceiling_absorption_f", {})
                self.wall_absorption_x_f = getattr(self, "wall_absorption_x_f", {})
                self.wall_absorption_y_f = getattr(self, "wall_absorption_y_f", {})

                self.floor_absorption_f[f] = self.floor_absorption
                self.ceiling_absorption_f[f] = self.ceiling_absorption
                self.wall_absorption_x_f[f] = self.wall_absorption_x
                self.wall_absorption_y_f[f] = self.wall_absorption_y

            # frequency response микрофона оставляем как было
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

        # небольшой случайный сдвиг источника (robustness), до ±5 см
        jitter_xy = 0.05  # м
        jitter_z = 0.05   # м

        x += float(self.rng.uniform(-jitter_xy, jitter_xy))
        y += float(self.rng.uniform(-jitter_xy, jitter_xy))
        z += float(self.rng.uniform(-jitter_z, jitter_z))

        # ограничим внутри комнаты с отступами
        margin = self.cfg.sources.margin_xy
        x = np.clip(x, margin, self.room_length - margin)
        y = np.clip(y, margin, self.room_width - margin)
        z = np.clip(z, self.cfg.sources.z_min, self.cfg.sources.z_max)

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

    def _generate_image_sources(
        self,
        x_src: float,
        y_src: float,
        z_src: float,
        frequency: float,
    ) -> List[Tuple[float, float, float, float]]:
        """
        Генерирует image sources до порядка self.max_image_order для прямоугольной комнаты.

        Возвращает список кортежей:
        (x_img, y_img, z_img, R_total)

        где R_total — совокупный коэффициент отражения для данного набора отражений.

        Алгоритм:
            - Для каждого направления x,y,z берём n_x, n_y, n_z в диапазоне [-order, order]
            - (n_x, n_y, n_z) = (0,0,0) — реальный источник (обработаем отдельно)
            - Для чётного n: x_img = x_src + 2*n*Lx
            Для нечётного n: x_img = -x_src + (2*n+2)*Lx
            (классическая формула image-source)
            - Аналогично для y,z.
            - Коэффициент отражения R_total примерно равен:
                R_floor^kz_floor * R_ceil^kz_ceil * R_x^kx * R_y^ky,
                где k* — количество пересечений соответствующих плоскостей.
            - Здесь для простоты: считаем, что каждое ненулевое n_x добавляет одно R_x, и т.п.
        """
        order = self.max_image_order
        Lx = self.room_length
        Ly = self.room_width
        Lz = self.room_height

        f = float(frequency)
        R_floor_f = 1.0 - self.floor_absorption_f[f]
        R_ceil_f = 1.0 - self.ceiling_absorption_f[f]
        R_x_f = 1.0 - self.wall_absorption_x_f[f]
        R_y_f = 1.0 - self.wall_absorption_y_f[f]

        image_sources: List[Tuple[float, float, float, float]] = []

        for nx in range(-order, order + 1):
            for ny in range(-order, order + 1):
                for nz in range(-order, order + 1):
                    if nx == 0 and ny == 0 and nz == 0:
                        continue  # реальный источник, не image

                    # формула image-source вдоль x
                    if nx % 2 == 0:
                        x_img = x_src + 2 * nx * Lx
                    else:
                        x_img = -x_src + (2 * nx + 2) * Lx

                    # вдоль y
                    if ny % 2 == 0:
                        y_img = y_src + 2 * ny * Ly
                    else:
                        y_img = -y_src + (2 * ny + 2) * Ly

                    # вдоль z (между 0 и Lz)
                    if nz % 2 == 0:
                        z_img = z_src + 2 * nz * Lz
                    else:
                        z_img = -z_src + (2 * nz + 2) * Lz

                    # грубая оценка числа отражений:
                    kx = abs(nx)
                    ky = abs(ny)
                    kz = abs(nz)

                    # считаем, что отражения по z поровну делятся между полом и потолком
                    # (можно усложнить, но для order<=2 это достаточно)
                    Rz = (R_floor_f * R_ceil_f) ** (kz / 2.0)

                    R_total = (R_x_f ** kx) * (R_y_f ** ky) * Rz
                    image_sources.append((x_img, y_img, z_img, R_total))

        return image_sources

    def _compute_pressure_field(self, sources: List[Dict[str, Any]], frequency: float) -> np.ndarray:
        """
        Поле от всех источников с учётом прямого пути и image sources до порядка self.max_image_order.
        """
        pressure = np.zeros((self.grid_x, self.grid_y), dtype=np.complex128)
        frequency = float(frequency)
        k = self.wave_numbers[frequency]
        alpha_air = self.absorption_coeffs[frequency]

        for src in sources:
            x_src = src["position"]["x"]
            y_src = src["position"]["y"]
            z_src = src["position"]["z"]

            amp = src["spectrum"].get(frequency, src["amplitude"] * 0.1)
            Q = amp * np.exp(1j * src["phase"])

            # ---------- прямой путь ----------
            r_direct = np.sqrt(
                (self.X - x_src) ** 2
                + (self.Y - y_src) ** 2
                + (self.Z - z_src) ** 2
            )
            r_direct = np.maximum(r_direct, 0.01)
            G_total = (
                np.exp(1j * k * r_direct)
                / (4.0 * np.pi * r_direct)
                * np.exp(-alpha_air * r_direct)
            )

            # ---------- image sources до порядка self.max_image_order ----------
            image_sources = self._generate_image_sources(x_src, y_src, z_src, frequency)

            for (x_img, y_img, z_img, R_total) in image_sources:
                r_img = np.sqrt(
                    (self.X - x_img) ** 2
                    + (self.Y - y_img) ** 2
                    + (self.Z - z_img) ** 2
                )
                r_img = np.maximum(r_img, 0.01)
                G_img = (
                    R_total
                    * np.exp(1j * k * r_img)
                    / (4.0 * np.pi * r_img)
                    * np.exp(-alpha_air * r_img)
                )
                G_total += G_img

            pressure += Q * G_total

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
