from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

import h5py
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from skimage.feature import peak_local_max
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import AcousticH5Dataset
from .models import build_model


@dataclass
class EvalConfig:
    model_name: str
    in_channels: int
    base_channels: int
    input_repr: str           # "complex" или "magnitude"
    ckpt_path: str
    test_h5: str
    dx: float                 # шаг сетки по x, м (0.039)
    dy: float                 # шаг сетки по y, м (0.031)
    distance_thresh_m: float  # 0.2 м
    peak_min_distance_px: int # 5
    peak_threshold_abs: float # 0.4
    device: str


def _load_model(cfg: EvalConfig) -> torch.nn.Module:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = build_model(
        name=cfg.model_name,
        in_channels=cfg.in_channels,
        out_channels=1,
        base_channels=cfg.base_channels,
    ).to(device)

    ckpt = torch.load(cfg.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def _detect_peaks(prob_map: np.ndarray, cfg: EvalConfig) -> np.ndarray:
    """prob_map: (H, W) numpy, в [0,1]. Возвращает массив [n_peaks, 2] (row, col)."""
    coordinates = peak_local_max(
        prob_map,
        min_distance=cfg.peak_min_distance_px,
        threshold_abs=cfg.peak_threshold_abs,
        exclude_border=False,
    )
    return coordinates


def _match_peaks(
    pred_peaks_rc: np.ndarray,
    gt_peaks_rc: np.ndarray,
    cfg: EvalConfig,
) -> Tuple[int, int, int, List[float]]:
    """
    Матчит предсказанные и истинные пики Венгерским алгоритмом.
    Возвращает TP, FP, FN и список расстояний (м) по TP.
    """
    if pred_peaks_rc.shape[0] == 0 and gt_peaks_rc.shape[0] == 0:
        return 0, 0, 0, []
    if pred_peaks_rc.shape[0] == 0:
        return 0, 0, int(gt_peaks_rc.shape[0]), []
    if gt_peaks_rc.shape[0] == 0:
        return 0, int(pred_peaks_rc.shape[0]), 0, []

    # перевод индексов (row, col) -> координаты в метрах
    pred_xy = np.stack(
        [pred_peaks_rc[:, 1] * cfg.dx, pred_peaks_rc[:, 0] * cfg.dy], axis=1
    )  # (N_pred, 2)
    gt_xy = np.stack(
        [gt_peaks_rc[:, 1] * cfg.dx, gt_peaks_rc[:, 0] * cfg.dy], axis=1
    )  # (N_gt, 2)

    # матрица расстояний
    dists = np.linalg.norm(pred_xy[:, None, :] - gt_xy[None, :, :], axis=-1)
    row_ind, col_ind = linear_sum_assignment(dists)

    tp = 0
    d_tp: List[float] = []
    used_gt = set()
    used_pred = set()

    for r, c in zip(row_ind, col_ind):
        if dists[r, c] <= cfg.distance_thresh_m:
            tp += 1
            d_tp.append(float(dists[r, c]))
            used_pred.add(r)
            used_gt.add(c)

    fp = pred_peaks_rc.shape[0] - len(used_pred)
    fn = gt_peaks_rc.shape[0] - len(used_gt)
    return tp, fp, fn, d_tp


def evaluate_model(cfg: EvalConfig) -> Dict[str, Any]:
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    model = _load_model(cfg)

    ds = AcousticH5Dataset(cfg.test_h5, input_repr=cfg.input_repr)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    # Загрузим также JSON/мету, если хочешь работать с истинными координатами.
    # Здесь считаем, что в HDF5 есть только source_maps (как в статье).

    total_tp = total_fp = total_fn = 0
    all_dists: List[float] = []

    with torch.no_grad():
        for x, y in tqdm(dl, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)  # (1, 1, H, W)
            prob_map = y_pred[0, 0].cpu().numpy()
            gt_map = y[0, 0].cpu().numpy()

            # предсказанные пики
            pred_peaks = _detect_peaks(prob_map, cfg)

            # истинные пики: ищем максимум на gauss-карте — в ноутбуке
            # у тебя, скорее всего, есть отдельные GT-координаты.
            # Здесь используем peak_local_max по высокой пороговой величине.
            # Если есть JSON-координаты, их можно сюда подставить вместо этого.
            gt_peaks = peak_local_max(
                gt_map,
                min_distance=cfg.peak_min_distance_px,
                threshold_abs=0.5,  # т.к. GT карта нормирована на 1
                exclude_border=False,
            )

            tp, fp, fn, d_tp = _match_peaks(pred_peaks, gt_peaks, cfg)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            all_dists.extend(d_tp)

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    if all_dists:
        mle_mean = float(np.mean(all_dists))
        mle_std = float(np.std(all_dists))
    else:
        mle_mean, mle_std = float("nan"), float("nan")

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mle_mean": mle_mean,
        "mle_std": mle_std,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "n_scenes": len(ds),
    }

    print(
        f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, "
        f"MLE={mle_mean:.3f} ± {mle_std:.3f} м"
    )

    ds.close()
    return metrics
