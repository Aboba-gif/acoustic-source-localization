from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import json
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AcousticH5Dataset
from .models import build_model


@dataclass
class EvalConfig:
    """
    Конфигурация для оценки модели.
    """
    model_name: str          # "unet_complex" / "unet_magnitude" / "shallow_cnn"
    in_channels: int         # 2 для комплексного входа, 1 для |p|
    base_channels: int       # сейчас не используется, оставлен для совместимости
    input_repr: str          # "complex" или "magnitude"
    ckpt_path: str           # путь к чекпойнту (models/.../best.ckpt)
    test_h5: str             # путь к HDF5 с тестом
    dx: float                # шаг сетки по x (м) -- бэкап, если нет JSON
    dy: float                # шаг сетки по y (м)
    distance_thresh_m: float # порог по расстоянию в метрах (0.2)
    peak_min_distance_px: int  # min_distance в пикселях (5)
    peak_threshold_abs: float  # threshold_abs для peak_local_max (на НОРМ. карте)
    device: str              # "cuda" или "cpu"


# =======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =======================

def _load_model(cfg: EvalConfig) -> torch.nn.Module:
    """
    Загружает модель и веса из чекпойнта.
    Чекпойнт ожидается в формате:
    {
      "model_state_dict": ...,
      ...
    }
    как сохраняет train.py.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(
        name=cfg.model_name,
        in_channels=cfg.in_channels,
        out_channels=1,
        base_channels=cfg.base_channels,
    ).to(device)

    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _find_peaks(heatmap: np.ndarray, min_distance: int, threshold_abs: float) -> np.ndarray:
    """
    Обёртка над peak_local_max:
    - heatmap: (H, W), значения в [0, 1].
    - возвращает массив координат в формате [N, 2]: (row, col).
    """
    return peak_local_max(
        heatmap,
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        exclude_border=False,
    )


def _get_true_sources_info(sample_id: int, json_root: Path) -> Dict[str, Any]:
    """
    Загружает JSON с описанием сцены по sample_id:
    data/metadata/json/sample_000001_info.json
    """
    json_path = json_root / f"sample_{int(sample_id):06d}_info.json"
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _convert_real_to_pixel(
    coords: Tuple[float, float],
    grid_dims: Tuple[int, int],
    room_dims: Tuple[float, float],
) -> Tuple[int, int]:
    """
    Переводит координаты (x,y) в метрах в (px_x, px_y) в индексах сетки.
    """
    x, y = coords
    grid_x, grid_y = grid_dims
    room_length, room_width = room_dims

    px = int(x / room_length * grid_x)
    py = int(y / room_width * grid_y)
    return px, py


def _validate_predictions_with_distances(
    predicted_peaks_rc: np.ndarray,
    true_sources: List[Dict[str, Any]],
    pixel_radius_threshold: float,
    grid_dims: Tuple[int, int],
    room_dims: Tuple[float, float],
) -> Tuple[Dict[str, Any], List[float]]:
    """
    Оценивает одну сцену:
      - predicted_peaks_rc: [N_pred, 2] (row, col) в пикселях
      - true_sources: список словарей с ['position']['x/y'] в метрах

    Возвращает:
      - словарь с метриками для сцены,
      - список расстояний (в пикселях) только для TP.
    """
    num_true = len(true_sources)
    num_pred = predicted_peaks_rc.shape[0]

    # Крайние случаи
    if num_true == 0 and num_pred == 0:
        metrics = {
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        return metrics, []

    if num_true == 0 and num_pred > 0:
        metrics = {
            "precision": 0.0,
            "recall": 1.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": num_pred,
            "fn": 0,
        }
        return metrics, []

    if num_true > 0 and num_pred == 0:
        metrics = {
            "precision": 1.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": num_true,
        }
        return metrics, []

    # Истинные координаты источников в метрах -> в пиксели
    true_peaks_real = np.array(
        [[s["position"]["x"], s["position"]["y"]] for s in true_sources],
        dtype=np.float32,
    )
    true_peaks_px = np.array(
        [_convert_real_to_pixel(tuple(c), grid_dims, room_dims) for c in true_peaks_real],
        dtype=np.float32,
    )

    # predicted_peaks_rc: (row, col) -> (x_px, y_px) = (col, row)
    pred_xy = np.stack(
        [predicted_peaks_rc[:, 1], predicted_peaks_rc[:, 0]],
        axis=1,
    )  # (N_pred, 2): (px_x, px_y)

    # Матрица расстояний в пикселях
    cost_matrix = cdist(pred_xy, true_peaks_px)
    pred_indices, true_indices = linear_sum_assignment(cost_matrix)

    distances = cost_matrix[pred_indices, true_indices]
    matched_mask = distances < pixel_radius_threshold

    true_positives = int(np.sum(matched_mask))
    false_positives = int(num_pred - true_positives)
    false_negatives = int(num_true - true_positives)

    precision = true_positives / (true_positives + false_positives + 1e-9)
    recall = true_positives / (true_positives + false_negatives + 1e-9)
    f1_score = (
        2.0 * (precision * recall) / (precision + recall + 1e-9)
        if (precision + recall) > 0.0
        else 0.0
    )

    matched_distances_px = distances[matched_mask]

    metrics = {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
    }
    return metrics, matched_distances_px.tolist()


# =======================
# ОСНОВНАЯ ФУНКЦИЯ ОЦЕНКИ
# =======================

def evaluate_model(cfg: EvalConfig) -> Dict[str, Any]:
    """
    Оценка модели на тестовой выборке.

    Шаги:
      - предсказываем карты model(x) (Sigmoid уже внутри моделей),
      - нормируем каждую карту по максимуму (0..1),
      - ищем пики через peak_local_max,
      - GT-координаты берём из JSON (x,y в метрах),
      - считаем TP/FP/FN через Венгерский алгоритм и порог по расстоянию.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg)

    ds = AcousticH5Dataset(cfg.test_h5, input_repr=cfg.input_repr)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Пути к JSON
    h5_path = Path(cfg.test_h5)
    json_root = h5_path.parent.parent / "metadata" / "json"

    # Габариты сетки
    grid_x = ds.pr_real.shape[1]
    grid_y = ds.pr_real.shape[2]

    # --- Получаем реальные размеры комнаты из первого JSON ---
    # Если по какой-то причине JSON не найдётся, fallback к cfg.dx, cfg.dy.
    try:
        # берём sample_id первой сцены
        first_sid = int(ds.sample_ids[0]) if ds.sample_ids is not None else 1
        first_info = _get_true_sources_info(first_sid, json_root)
        room_info = first_info.get("room", {})
        room_length = float(room_info.get("Lx", cfg.dx * grid_x))
        room_width = float(room_info.get("Ly", cfg.dy * grid_y))
    except Exception:
        room_length = cfg.dx * grid_x
        room_width = cfg.dy * grid_y

    dx_eff = room_length / grid_x
    dy_eff = room_width / grid_y
    pixel_radius_threshold = cfg.distance_thresh_m / dx_eff

    scene_metrics: List[Dict[str, Any]] = []
    all_matched_distances_px: List[float] = []
    total_tp = total_fp = total_fn = 0

    with torch.no_grad():
        for batch in tqdm(dl, desc="Evaluating"):
            x, _y, sample_id_tensor = batch
            sample_id = int(sample_id_tensor.item())

            x = x.to(device)

            # Предсказанная карта [1,1,H,W] -> [H,W]
            y_pred = model(x)
            prob_map = y_pred[0, 0].cpu().numpy()

            # Нормировка по максимуму для стабильного порога
            if prob_map.max() > 0:
                prob_map_norm = prob_map / prob_map.max()
            else:
                prob_map_norm = prob_map

            # Поиск пиков на предсказанной карте
            predicted_peaks_rc = _find_peaks(
                prob_map_norm,
                min_distance=cfg.peak_min_distance_px,
                threshold_abs=cfg.peak_threshold_abs,  # теперь это порог на норм. карте
            )

            true_info = _get_true_sources_info(sample_id, json_root)
            true_sources = true_info.get("sources", [])

            metrics_i, matched_distances_px = _validate_predictions_with_distances(
                predicted_peaks_rc,
                true_sources,
                pixel_radius_threshold=pixel_radius_threshold,
                grid_dims=(grid_x, grid_y),
                room_dims=(room_length, room_width),
            )

            scene_metrics.append(metrics_i)
            all_matched_distances_px.extend(matched_distances_px)
            total_tp += metrics_i["tp"]
            total_fp += metrics_i["fp"]
            total_fn += metrics_i["fn"]

    # --- Агрегация метрик ---
    # Микро-усреднение по суммарным TP/FP/FN
    precision_micro = total_tp / (total_tp + total_fp + 1e-9)
    recall_micro = total_tp / (total_tp + total_fn + 1e-9)
    f1_micro = (
        2.0 * precision_micro * recall_micro
        / (precision_micro + recall_micro + 1e-9)
    )

    # Дополнительно можно сохранить средний F1 по сценам (не обязателен)
    f1s = [m["f1_score"] for m in scene_metrics]
    mean_f1_per_scene = float(np.mean(f1s)) if f1s else float("nan")

    all_matched_distances_px_arr = np.array(all_matched_distances_px, dtype=np.float32)
    if all_matched_distances_px_arr.size > 0:
        all_matched_distances_m = all_matched_distances_px_arr * dx_eff
        mle_mean = float(np.mean(all_matched_distances_m))
        mle_std = float(np.std(all_matched_distances_m))
    else:
        mle_mean = float("nan")
        mle_std = float("nan")

    metrics = {
        "precision": float(precision_micro),
        "recall": float(recall_micro),
        "f1": float(f1_micro),
        "f1_mean_per_scene": mean_f1_per_scene,
        "mle_mean": mle_mean,
        "mle_std": mle_std,
        "n_scenes": len(ds),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
    }

    print(
        f"Precision={precision_micro:.3f}, Recall={recall_micro:.3f}, "
        f"F1={f1_micro:.3f}, MLE={mle_mean:.3f} ± {mle_std:.3f} м, "
        f"TP={total_tp}, FP={total_fp}, FN={total_fn}"
    )

    ds.close()
    return metrics
