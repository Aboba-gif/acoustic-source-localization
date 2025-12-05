from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, Any

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max


# =======================
# ВСПОМОГАТЕЛЬНЫЕ СТРУКТУРЫ
# =======================

@dataclass
class GridMeta:
    """
    Метаданные сетки и комнаты, используются для конвертации координат
    и подписей осей в визуализациях.
    """
    nx: int
    ny: int
    dx: float  # шаг сетки по x (м)
    dy: float  # шаг сетки по y (м)

    @property
    def Lx(self) -> float:
        return self.nx * self.dx

    @property
    def Ly(self) -> float:
        return self.ny * self.dy


# =======================
# ЗАГРУЗКА H5 + JSON
# =======================

def find_sample_in_h5(
    h5_path: str | Path,
    sample_id: int,
) -> int:
    """
    Находит индекс сцены с данным sample_id внутри HDF5.

    Если в файле есть датасет 'sample_ids', ищем по нему.
    Если нет — считаем, что sample_id == idx + 1.
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        n = f["pressure_real"].shape[0]
        if "sample_ids" in f:
            sample_ids = f["sample_ids"][:]
            matches = np.where(sample_ids == sample_id)[0]
            if len(matches) == 0:
                raise ValueError(f"sample_id={sample_id} not found in {h5_path}")
            return int(matches[0])
        else:
            if not (1 <= sample_id <= n):
                raise ValueError(
                    f"sample_id={sample_id} is out of range for dataset of size {n}"
                )
            return sample_id - 1


def load_scene_from_h5(
    h5_path: str | Path,
    index: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Загружает одну сцену из H5 по индексу.

    Возвращает:
      - p_real:  (H, W)
      - p_imag:  (H, W)
      - s_map:   (H, W)
      - sample_id: int
    """
    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as f:
        p_real = f["pressure_real"][index].astype(np.float32)
        p_imag = f["pressure_imag"][index].astype(np.float32)
        s_map = f["source_maps"][index].astype(np.float32)
        if "sample_ids" in f:
            sample_id = int(f["sample_ids"][index])
        else:
            sample_id = index + 1
    return p_real, p_imag, s_map, sample_id


def load_scene_metadata(
    json_root: str | Path,
    sample_id: int,
) -> Dict[str, Any]:
    """
    Загружает JSON с описанием сцены:
      json_root / f"sample_{sample_id:06d}_info.json"
    """
    json_root = Path(json_root)
    json_path = json_root / f"sample_{sample_id:06d}_info.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# =======================
# ПОИСК ПИКОВ
# =======================

def find_peaks_on_map(
    heatmap: np.ndarray,
    min_distance: int = 5,
    threshold_abs: float = 0.4,
) -> np.ndarray:
    """
    Обёртка над skimage.feature.peak_local_max
    для карт (H, W) в [0,1].

    Возвращает массив [N, 2] координат (row, col).
    """
    return peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False,
    )


# =======================
# ВИЗУАЛИЗАТОР СЫРЫХ СЭМПЛОВ
# =======================

def visualize_dataset_sample(
    project_dir: str,
    sample_id: int,
    split: str | None = None,
    cmap_field: str = "viridis",
    cmap_phase: str = "hsv",
    cmap_smap: str = "hot",
    figsize: tuple[int, int] = (18, 6),
) -> None:
    """
    ОТДЕЛЬНЫЙ визуализатор сырых сэмплов из датасета (без моделей).

    Показывает:
      1) Амплитуду комплексного поля |p(x,y)|
      2) Фазу поля arg p(x,y)
      3) GT-карту источников (source_map) с истинными координатами (в метрах).
    """
    project_dir = Path(project_dir)
    data_dir = project_dir / "data"
    h5_dir = data_dir / "processed"
    json_dir = data_dir / "metadata" / "json"

    # --- найти split и индекс ---
    splits_to_check = [split] if split is not None else ["train", "val", "test"]
    found = None

    for sp in splits_to_check:
        h5_path = h5_dir / f"{sp}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            if "sample_ids" not in f:
                continue
            sample_ids = f["sample_ids"][:]
            matches = np.where(sample_ids == sample_id)[0]
            if len(matches) > 0:
                found = (sp, h5_path, int(matches[0]))
                break

    if found is None:
        print(f"Сэмпл ID={sample_id} не найден ни в train/val/test H5.")
        return

    split_name, h5_path, index_in_split = found
    print(f"Сэмпл {sample_id} найден в '{split_name}.h5' по индексу {index_in_split}.")

    # --- данные из H5 ---
    with h5py.File(h5_path, "r") as f:
        p_real = f["pressure_real"][index_in_split].astype(np.float32)
        p_imag = f["pressure_imag"][index_in_split].astype(np.float32)
        s_map = f["source_maps"][index_in_split].astype(np.float32)

    pr_noisy = p_real + 1j * p_imag

    # --- JSON с GT ---
    json_path = json_dir / f"sample_{sample_id:06d}_info.json"
    if not json_path.exists():
        print(f"Файл с ground truth не найден: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        info = json.load(f)

    room = info.get("room", {})
    room_length = room.get("Lx", p_real.shape[0])
    room_width = room.get("Ly", p_real.shape[1])
    plot_extent = [0, room_length, 0, room_width]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) Амплитуда |p|
    mag = np.abs(pr_noisy).T  # .T: (x,y)->(col,row)
    if mag.max() > 0:
        mag = mag / mag.max()
    im1 = axes[0].imshow(mag, cmap=cmap_field, origin="lower", extent=plot_extent)
    axes[0].set_title("Noisy Pressure Field (Amplitude)")
    axes[0].set_xlabel("X (m)")
    axes[0].set_ylabel("Y (m)")
    plt.colorbar(im1, ax=axes[0])

    # 2) Фаза arg(p)
    phase = np.angle(pr_noisy).T
    im2 = axes[1].imshow(
        phase,
        cmap=cmap_phase,
        vmin=-np.pi,
        vmax=np.pi,
        origin="lower",
        extent=plot_extent,
    )
    axes[1].set_title("Noisy Pressure Field (Phase)")
    axes[1].set_xlabel("X (m)")
    axes[1].set_ylabel("Y (m)")
    plt.colorbar(im2, ax=axes[1])

    # 3) GT source map
    sm = s_map.T
    im3 = axes[2].imshow(sm, cmap=cmap_smap, origin="lower", extent=plot_extent)
    axes[2].set_title("Ground Truth Source Map")
    axes[2].set_xlabel("X (m)")
    axes[2].set_ylabel("Y (m)")
    plt.colorbar(im3, ax=axes[2])

    # GT источники в метрах
    for src in info.get("sources", []):
        x_pos = src["position"]["x"]
        y_pos = src["position"]["y"]
        for ax in axes:
            ax.plot(
                x_pos,
                y_pos,
                "c*",
                markersize=12,
                markeredgecolor="black",
                label="Ground Truth",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        by_label = dict(zip(labels, handles))
        axes[0].legend(by_label.values(), by_label.keys())

    plt.suptitle(
        f"Sample {sample_id} | split: {split_name} | "
        f"Frequency: {info['frequency']} Hz | Sources: {len(info['sources'])}",
        fontsize=16,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# =======================
# ВИЗУАЛИЗАТОР ДЛЯ МОДЕЛЕЙ (триплет)
# =======================

def visualize_sample_triplet(
    h5_path: str | Path,
    model_pred_map: np.ndarray,
    sample_index: int | None = None,
    sample_id: int | None = None,
    json_root: str | Path | None = None,
    peak_min_distance_px: int = 5,
    peak_threshold_abs_pred: float = 0.4,
    peak_threshold_abs_gt: float = 0.5,
    figsize: Tuple[int, int] = (18, 6),
    suptitle: str | None = None,
    cmap_field: str = "magma",
    cmap_prob: str = "magma",
) -> plt.Figure:
    """
    Тройная визуализация для одной сцены (как в примере для статьи):

      A) Истинная карта источников (GT source_map)
      B) Предсказанная карта (model_pred_map, [H,W] в [0,1])
      C) Модуль поля |p(x,y)| с:
         - зелёными крестами в истинных позициях источников (из JSON),
         - красными квадратиками в позициях пиков предсказанной карты.

    ВСЁ рисуем в ИНДЕКСАХ:
      - imshow(... .T, origin='lower')
      - пики и кресты в тех же координатах, что и картинка.
    """
    h5_path = Path(h5_path)

    # --- индекс / sample_id ---
    if sample_index is None:
        if sample_id is None:
            raise ValueError("Нужно указать либо sample_index, либо sample_id.")
        sample_index = find_sample_in_h5(h5_path, sample_id)

    p_real, p_imag, s_map, sid = load_scene_from_h5(h5_path, sample_index)
    if sample_id is None:
        sample_id = sid

    H, W = s_map.shape
    if model_pred_map.shape != (H, W):
        raise ValueError(
            f"Размеры model_pred_map {model_pred_map.shape} не совпадают с GT {s_map.shape}"
        )

    # --- JSON c GT источниками ---
    if json_root is None:
        json_root = h5_path.parent.parent / "metadata" / "json"
    meta = load_scene_metadata(json_root, sample_id)
    true_sources = meta.get("sources", [])

    # --- пики на картах (в индексах исходных массивов: row, col) ---
    gt_peaks = find_peaks_on_map(
        s_map,
        min_distance=peak_min_distance_px,
        threshold_abs=peak_threshold_abs_gt,
    )
    pred_peaks = find_peaks_on_map(
        model_pred_map,
        min_distance=peak_min_distance_px,
        threshold_abs=peak_threshold_abs_pred,
    )

    # --- модуль комплексного поля ---
    mag = np.sqrt(p_real**2 + p_imag**2)
    mag_norm = mag / mag.max() if mag.max() > 0 else mag

    # --- фигура ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if suptitle is None:
        suptitle = f"Train sample #{sample_id} (GT as prediction stub)"
    fig.suptitle(suptitle, fontsize=16)

    # NB: ВСЕ imshow с .T, origin='lower' (ось 0 -> Y, ось 1 -> X)
    # ---- A) GT карта источников ----
    ax0 = axes[0]
    im0 = ax0.imshow(s_map.T, cmap=cmap_prob, origin="lower")
    ax0.set_title("A) Истинная карта источников")
    ax0.set_xlabel("X, пиксели")
    ax0.set_ylabel("Y, пиксели")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # ---- B) Предсказанная карта ----
    ax1 = axes[1]
    im1 = ax1.imshow(model_pred_map.T, cmap=cmap_prob, origin="lower", vmin=0.0, vmax=1.0)
    ax1.set_title("B) Предсказанная карта")
    ax1.set_xlabel("X, пиксели")
    ax1.set_ylabel("Y, пиксели")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ---- C) |p| + GT и предсказанные пики ----
    ax2 = axes[2]
    im2 = ax2.imshow(mag_norm.T, cmap=cmap_field, origin="lower")
    ax2.set_title("C) Результаты на поле |p(x, y)|")
    ax2.set_xlabel("X, пиксели")
    ax2.set_ylabel("Y, пиксели")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # --- истинные источники: метры -> индексы (как в генераторе), затем -> XY после .T ---
    true_x_px = []
    true_y_px = []

    # из json: x_m, y_m
    # в генераторе density_map[ ix, iy ], где ix = x/dx, iy = y/dy
    # но dx/dy у нас тут нет, поэтому используем масштабирование через Lx/Ly, nx/ny
    room = meta.get("room", {})
    Lx = room.get("Lx", 1.0)
    Ly = room.get("Ly", 1.0)
    nx, ny = H, W  # (!) axis0, axis1

    for s in true_sources:
        x_m = s["position"]["x"]
        y_m = s["position"]["y"]
        ix = x_m / Lx * nx  # индекс по оси 0
        iy = y_m / Ly * ny  # индекс по оси 1

        # после .T: axis0 -> Y, axis1 -> X
        x_px = iy
        y_px = ix
        true_x_px.append(x_px)
        true_y_px.append(y_px)

    if true_x_px:
        ax2.scatter(true_x_px, true_y_px, s=80, c="lime", marker="x", linewidths=2,
                    label="Истина")

    # --- предсказанные пики (row,col) -> (x=col, y=row) после .T ---
    if pred_peaks.size > 0:
        pred_row = pred_peaks[:, 0]
        pred_col = pred_peaks[:, 1]
        ax2.scatter(
            pred_col,
            pred_row,
            s=80,
            facecolors="none",
            edgecolors="red",
            linewidths=2,
            marker="s",
            label="Предсказание",
        )

    ax2.legend(loc="upper right", fontsize=10)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig


# =======================
# ПРОСТАЯ ОДИНОЧНАЯ КАРТА
# =======================

def visualize_single_map(
    data: np.ndarray,
    title: str = "",
    cmap: str = "magma",
    figsize: Tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Простая визуализация одной 2D-карты (например, предсказаний модели).
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(data, cmap=cmap, origin="lower")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X, пиксели")
    ax.set_ylabel("Y, пиксели")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
