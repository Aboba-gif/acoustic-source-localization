from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Tuple, List, Dict, Any

import h5py
import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
import torch

from .dataset import AcousticH5Dataset


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

    В generator.py мы сохраняем sample_ids в отдельный датасет.
    Если его нет — считаем, что sample_id == idx + 1.

    Возвращает:
      - индекс в диапазоне [0, N-1], по которому можно брать данные из H5.
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
            # fallback: порядок = sample_id - 1
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
# ВИЗУАЛИЗАЦИЯ
# =======================

def _setup_axes_labels(ax, grid_meta: GridMeta, title: str) -> None:
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("X, пиксели")
    ax.set_ylabel("Y, пиксели")
    # при желании можно добавить вторые оси в метрах — пока оставим в пикселях


def visualize_sample_triplet(
    h5_path: str | Path,
    model_pred_map: np.ndarray,
    sample_index: int | None = None,
    sample_id: int | None = None,
    json_root: str | Path | None = None,
    grid_dx: float | None = None,
    grid_dy: float | None = None,
    peak_min_distance_px: int = 5,
    peak_threshold_abs_pred: float = 0.4,
    peak_threshold_abs_gt: float = 0.5,
    figsize: Tuple[int, int] = (18, 6),
    suptitle: str | None = None,
    cmap_field: str = "magma",
    cmap_prob: str = "magma",
) -> plt.Figure:
    """
    Строит тройной рисунок для одной сцены (аналог того, что ты показывал в ноутбуке):

      A) Истинная карта источников (GT source_map)
      B) Предсказанная карта (model_pred_map)
      C) Модуль поля |p(x,y)| с отмеченными источниками:
         - зелёные кресты: истинные источники
         - красные квадраты: предсказанные пики

    Параметры:
      - h5_path: путь к HDF5 (train/val/test).
      - model_pred_map: 2D-массив (H,W) с предсказанной картой
        (уже после Sigmoid, в [0,1]).
      - sample_index: индекс в H5 (если известен).
      - sample_id: sample_id (если хочешь привязаться к JSON);
        если задан только sample_id, индекс будет найден автоматически.
      - json_root: корень с JSON (по умолчанию data/metadata/json,
        выведется из h5_path).
      - grid_dx, grid_dy: шаги сетки в метрах (если нужны для
        вычисления/подписей). Можно не указывать — отобразим только пиксели.
    """
    h5_path = Path(h5_path)

    if sample_index is None:
        if sample_id is None:
            raise ValueError("Нужно указать либо sample_index, либо sample_id.")
        sample_index = find_sample_in_h5(h5_path, sample_id)
    # если sample_id не задан, восстановим его из H5
    p_real, p_imag, s_map, sid = load_scene_from_h5(h5_path, sample_index)
    if sample_id is None:
        sample_id = sid

    H, W = s_map.shape
    if model_pred_map.shape != (H, W):
        raise ValueError(
            f"Размеры model_pred_map {model_pred_map.shape} не совпадают с GT {s_map.shape}"
        )

    # grid meta (если есть dx/dy)
    if grid_dx is not None and grid_dy is not None:
        grid_meta = GridMeta(nx=H, ny=W, dx=grid_dx, dy=grid_dy)
    else:
        # заглушка только для подписей, без реальных размеров
        grid_meta = GridMeta(nx=H, ny=W, dx=1.0, dy=1.0)

    # JSON с истинными источниками
    if json_root is None:
        # предполагаем структуру data/processed/ -> data/metadata/json
        json_root = h5_path.parent.parent / "metadata" / "json"
    meta = load_scene_metadata(json_root, sample_id)
    true_sources = meta.get("sources", [])

    # Истинные пики на карте плотности
    gt_peaks = find_peaks_on_map(
        s_map,
        min_distance=peak_min_distance_px,
        threshold_abs=peak_threshold_abs_gt,
    )

    # Предсказанные пики на предсказанной карте
    pred_peaks = find_peaks_on_map(
        model_pred_map,
        min_distance=peak_min_distance_px,
        threshold_abs=peak_threshold_abs_pred,
    )

    # Модуль комплексного поля
    mag = np.sqrt(p_real**2 + p_imag**2)
    if mag.max() > 0:
        mag_norm = mag / mag.max()
    else:
        mag_norm = mag

    # Фигура
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    if suptitle is None:
        suptitle = f"Результат для сэмпла #{sample_id}"
    fig.suptitle(suptitle, fontsize=16)

    # ---- A) GT карта источников ----
    ax0 = axes[0]
    # как в ноутбуке: показываем транспонированную карту
    im0 = ax0.imshow(s_map.T, cmap=cmap_prob, origin="lower")
    _setup_axes_labels(ax0, grid_meta, "A) Истинная карта источников")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # ---- B) Предсказанная карта ----
    ax1 = axes[1]
    im1 = ax1.imshow(model_pred_map.T, cmap=cmap_prob, origin="lower", vmin=0.0, vmax=1.0)
    _setup_axes_labels(ax1, grid_meta, "B) Предсказанная карта")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # ---- C) Модуль поля |p| + GT и предсказанные пики ----
    ax2 = axes[2]
    im2 = ax2.imshow(mag_norm.T, cmap=cmap_field, origin="lower")
    _setup_axes_labels(ax2, grid_meta, "C) Результаты на поле |p(x, y)|")

    # Истинные позиции (в метрах) → пиксели
    true_x = []
    true_y = []
    for s in true_sources:
        x_m = s["position"]["x"]
        y_m = s["position"]["y"]
        # чтобы картинка совпала с ноутбуком, используем ту же формулу, но
        # учитываем, что мы показываем .T: оси меняются местами
        px = int(x_m / grid_meta.Lx * grid_meta.nx)
        py = int(y_m / grid_meta.Ly * grid_meta.ny)
        true_x.append(px)
        true_y.append(py)

    # Зелёные кресты — истина (используем (px, py) без перестановки:
    # в картинке после .T это будет как в ноутбуке)
    if len(true_x) > 0:
        ax2.scatter(true_x, true_y, s=60, c="lime", marker="x", label="Истина")

    # Красные квадраты — предсказанные пики
    # В ноутбуке scatter(true_peaks[:,0], true_peaks[:,1]) с картой .T,
    # поэтому здесь тоже НЕ меняем порядок координат.
    if pred_peaks.size > 0:
        ax2.scatter(
            pred_peaks[:, 0],  # x
            pred_peaks[:, 1],  # y
            s=60,
            facecolors="none",
            edgecolors="red",
            linewidths=1.5,
            marker="s",
            label="Предсказание",
        )

    ax2.legend(loc="upper right", fontsize=10)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def visualize_sample_legacy_article(
    dataset: AcousticH5Dataset,
    model: torch.nn.Module,
    device: torch.device,
    index: int,
    peak_min_distance_px: int = 5,
    peak_threshold_abs_pred: float = 0.4,
    peak_threshold_abs_gt: float = 0.5,
    figsize: Tuple[int, int] = (18, 5.5),
    outputs_dir: str | Path = "results/figures",
) -> Path:
    """
    Точная копия visualize_sample_for_article из старого Colab.
    Нужна только для воспроизведения рисунков статьи 1-в-1.
    """
    import os
    from .evaluate import _find_peaks as find_peaks  # тот же peak_local_max

    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # --- Берём вход и GT так же, как в ноутбуке ---
    input_tensor, target_tensor, sample_id = dataset[index]
    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.cpu().numpy()  # (1, H, W)
    true_map = target_tensor.squeeze()           # (H, W)

    # --- Предсказание ---
    with torch.no_grad():
        pred_map = model(input_tensor.unsqueeze(0)).squeeze().cpu().numpy()  # (H, W)

    # --- Пики как в ноутбуке: на немасштабированных картах ---
    predicted_peaks = find_peaks(pred_map, peak_min_distance_px, peak_threshold_abs_pred)
    true_peaks = find_peaks(true_map, peak_min_distance_px, peak_threshold_abs_gt)

    # --- Амплитуда входного поля ---
    input_amplitude = np.sqrt(input_tensor[0].cpu().numpy()**2 +
                              input_tensor[1].cpu().numpy()**2)

    # --- Фигура из трёх панелей с .T, как в Colab ---
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # 1) GT
    axes[0].imshow(true_map.T, cmap="hot", origin="lower")
    axes[0].set_title("A) Истинная карта источников")

    # 2) Prediction
    axes[1].imshow(pred_map.T, cmap="hot", origin="lower", vmin=0, vmax=1)
    axes[1].set_title("B) Предсказанная карта (Выход U-Net)")

    # 3) Overlay
    axes[2].imshow(input_amplitude.T, cmap="viridis", origin="lower", alpha=0.8)
    axes[2].set_title("C) Результаты на поле давления")
    axes[2].scatter(true_peaks[:, 0], true_peaks[:, 1],
                    c="lime", marker="x", s=150, linewidth=2, label="Истина")
    axes[2].scatter(predicted_peaks[:, 0], predicted_peaks[:, 1],
                    edgecolor="red", facecolor="none",
                    marker="s", s=150, linewidth=2, label="Предсказание")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("X, пиксели")
        ax.set_ylabel("Y, пиксели")

    fig.suptitle(f"Результат для сэмпла #{sample_id}", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = outputs_dir / f"article_visualization_sample_{int(sample_id)}.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path



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
