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
    dx: float                # шаг сетки по x (м)
    dy: float                # шаг сетки по y (м)
    distance_thresh_m: float # порог по расстоянию в метрах (0.2)
    peak_min_distance_px: int  # min_distance в пикселях (5)
    peak_threshold_abs: float  # threshold_abs для peak_local_max (0.4 для NN)
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

    В Colab:
        px = int(x / room_length * grid_x)
        py = int(y / room_width  * grid_y)
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
    Логика строго как в старом ноутбуке:
    - predicted_peaks_rc трактуем так же, как раньше: как массив (row, col),
      без перестановки осей;
    - true_sources -> пиксели через convert_real_to_pixel;
    - расстояния считаем cdist(predicted_peaks_rc, true_peaks_px).

    Возвращаем:
      - словарь с precision/recall/f1 и TP/FP/FN,
      - список расстояний (в пикселях) только для TP.
    """
    num_true = len(true_sources)
    num_pred = predicted_peaks_rc.shape[0]

    # Крайние случаи — как в старом validate_predictions_with_distances
    if num_true == 0:
        metrics = {
            "precision": 1.0 if num_pred == 0 else 0.0,
            "recall": 1.0,
            "f1_score": 1.0 if num_pred == 0 else 0.0,
            "tp": 0,
            "fp": num_pred if num_pred > 0 else 0,
            "fn": 0,
        }
        return metrics, []

    if num_pred == 0:
        metrics = {
            "precision": 1.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": num_true,
        }
        return metrics, []

    # Истина в метрах -> в пиксели
    true_peaks_real = np.array(
        [[s["position"]["x"], s["position"]["y"]] for s in true_sources],
        dtype=np.float32,
    )
    true_peaks_px = np.array(
        [_convert_real_to_pixel(tuple(c), grid_dims, room_dims)
         for c in true_peaks_real],
        dtype=np.float32,
    )

    # ВАЖНО: здесь мы НЕ меняем порядок координат predicted_peaks_rc,
    # используем их как в старом коде: (row, col) прямо в cdist.
    cost_matrix = cdist(predicted_peaks_rc.astype(np.float32), true_peaks_px)
    pred_indices, true_indices = linear_sum_assignment(cost_matrix)

    distances = cost_matrix[pred_indices, true_indices]
    matched_mask = distances < pixel_radius_threshold
    true_positives = int(np.sum(matched_mask))

    precision = true_positives / num_pred
    recall = true_positives / num_true
    f1_score = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    matched_distances_px = distances[matched_mask]

    # FN/FP для совместимости с новым кодом
    false_positives = int(num_pred - true_positives)
    false_negatives = int(num_true - true_positives)

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
    Основная функция оценки модели на тестовой выборке.
    Повторяет логику evaluate_and_generate_results из Colab (часть с метриками),
    но без генерации картинок — это делает scripts/evaluate_all.py.

    Использует:
      - предсказанные карты model(x) (Sigmoid уже внутри моделей),
      - пики через peak_local_max на выходе сети,
      - ground truth источники из JSON (sample_{id}_info.json),
      - Венгерский алгоритм и порог расстояния (в пикселях) для TP/FP/FN.
    """
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg)

    ds = AcousticH5Dataset(cfg.test_h5, input_repr=cfg.input_repr)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # пути к JSON
    h5_path = Path(cfg.test_h5)
    json_root = h5_path.parent.parent / "metadata" / "json"

    # Габариты сетки и комнаты
    grid_x = ds.pr_real.shape[1]
    grid_y = ds.pr_real.shape[2]
    room_length = cfg.dx * grid_x  # dx = room_length / grid_x
    room_width = cfg.dy * grid_y   # dy = room_width  / grid_y

    dx = cfg.dx
    pixel_radius_threshold = cfg.distance_thresh_m / dx

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

            # Поиск пиков на предсказанной карте
            predicted_peaks_rc = _find_peaks(
                prob_map,
                min_distance=cfg.peak_min_distance_px,
                threshold_abs=cfg.peak_threshold_abs,
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

    # Аггрегируем метрики по всей тестовой выборке (средние по сценам)
    precisions = [m["precision"] for m in scene_metrics]
    recalls = [m["recall"] for m in scene_metrics]
    f1s = [m["f1_score"] for m in scene_metrics]

    mean_precision = float(np.mean(precisions)) if precisions else float("nan")
    mean_recall = float(np.mean(recalls)) if recalls else float("nan")
    mean_f1 = float(np.mean(f1s)) if f1s else float("nan")

    all_matched_distances_px_arr = np.array(all_matched_distances_px, dtype=np.float32)
    if all_matched_distances_px_arr.size > 0:
        all_matched_distances_m = all_matched_distances_px_arr * dx
        mle_mean = float(np.mean(all_matched_distances_m))
        mle_std = float(np.std(all_matched_distances_m))
    else:
        mle_mean = float("nan")
        mle_std = float("nan")

    metrics = {
        "precision": mean_precision,
        "recall": mean_recall,
        "f1": mean_f1,
        "mle_mean": mle_mean,
        "mle_std": mle_std,
        "n_scenes": len(ds),
        "tp": int(total_tp),
        "fp": int(total_fp),
        "fn": int(total_fn),
    }

    print(
        f"Precision={mean_precision:.3f}, Recall={mean_recall:.3f}, "
        f"F1={mean_f1:.3f}, MLE={mle_mean:.3f} ± {mle_std:.3f} м, "
        f"TP={total_tp}, FP={total_fp}, FN={total_fn}"
    )

    ds.close()
    return metrics
