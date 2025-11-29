import argparse
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
import yaml
from skimage.feature import peak_local_max
from tqdm import tqdm

from acoustic_loc.evaluate import EvalConfig, evaluate_model
from acoustic_loc.dataset import AcousticH5Dataset


def evaluate_smoothed_amplitude(
    test_h5: str,
    dx: float,
    dy: float,
    distance_thresh_m: float,
    peak_min_distance_px: int,
    peak_threshold_abs: float,
) -> dict:
    """
    Бейзлайн: сглаженная амплитуда |p|.
    Реализуем здесь, не используя нейросети.
    """
    from scipy.ndimage import gaussian_filter
    from scipy.optimize import linear_sum_assignment

    h5_path = Path(test_h5)
    with h5py.File(h5_path, "r") as h5:
        pr_real = h5["pressure_real"]
        pr_imag = h5["pressure_imag"]
        smap = h5["source_maps"]

        n = pr_real.shape[0]

        total_tp = total_fp = total_fn = 0
        all_dists = []

        for i in tqdm(range(n), desc="Smoothed |p| baseline"):
            p_real = pr_real[i].astype(np.float32)
            p_imag = pr_imag[i].astype(np.float32)
            gt_map = smap[i].astype(np.float32)

            mag = np.sqrt(p_real ** 2 + p_imag ** 2)
            mag = mag / (mag.max() + 1e-9)

            mag_smooth = gaussian_filter(mag, sigma=3.0)

            pred_peaks = peak_local_max(
                mag_smooth,
                min_distance=peak_min_distance_px,
                threshold_abs=peak_threshold_abs,
                exclude_border=False,
            )

            gt_peaks = peak_local_max(
                gt_map,
                min_distance=peak_min_distance_px,
                threshold_abs=0.5,
                exclude_border=False,
            )

            if pred_peaks.shape[0] == 0 and gt_peaks.shape[0] == 0:
                continue

            if pred_peaks.shape[0] == 0:
                total_fn += int(gt_peaks.shape[0])
                continue

            if gt_peaks.shape[0] == 0:
                total_fp += int(pred_peaks.shape[0])
                continue

            pred_xy = np.stack(
                [pred_peaks[:, 1] * dx, pred_peaks[:, 0] * dy],
                axis=1,
            )
            gt_xy = np.stack(
                [gt_peaks[:, 1] * dx, gt_peaks[:, 0] * dy],
                axis=1,
            )

            dists = np.linalg.norm(
                pred_xy[:, None, :] - gt_xy[None, :, :],
                axis=-1,
            )
            row_ind, col_ind = linear_sum_assignment(dists)

            used_pred = set()
            used_gt = set()
            for r, c in zip(row_ind, col_ind):
                if dists[r, c] <= distance_thresh_m:
                    total_tp += 1
                    all_dists.append(float(dists[r, c]))
                    used_pred.add(r)
                    used_gt.add(c)

            total_fp += pred_peaks.shape[0] - len(used_pred)
            total_fn += gt_peaks.shape[0] - len(used_gt)

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    if all_dists:
        mle_mean = float(np.mean(all_dists))
        mle_std = float(np.std(all_dists))
    else:
        mle_mean = mle_std = float("nan")

    return {
        "method": "Smoothed |p|",
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mle_mean": mle_mean,
        "mle_std": mle_std,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument(
        "--out_csv",
        type=str,
        default="results/tables/metrics_baselines.csv",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    test_h5 = cfg["data"]["test_h5"]
    dx = cfg["grid"]["dx"]
    dy = cfg["grid"]["dy"]
    dist_thr = cfg["eval"]["distance_thresh_m"]
    min_dist_px = cfg["eval"]["peak_min_distance_px"]
    thr_nn = cfg["eval"]["peak_threshold_abs_nn"]
    thr_amp = cfg["eval"]["peak_threshold_abs_amp"]

    results = []

    # 1) Smoothed amplitude baseline
    res_smooth = evaluate_smoothed_amplitude(
        test_h5=test_h5,
        dx=dx,
        dy=dy,
        distance_thresh_m=dist_thr,
        peak_min_distance_px=min_dist_px,
        peak_threshold_abs=thr_amp,
    )
    results.append(res_smooth)

    # 2) Нейросети
    for name, mcfg in cfg["models"].items():
        print(f"\n=== Evaluating {name} ===")
        ecfg = EvalConfig(
            model_name=name,
            in_channels=mcfg["in_channels"],
            base_channels=mcfg["base_channels"],
            input_repr=mcfg["input_repr"],
            ckpt_path=mcfg["ckpt_path"],
            test_h5=test_h5,
            dx=dx,
            dy=dy,
            distance_thresh_m=dist_thr,
            peak_min_distance_px=min_dist_px,
            peak_threshold_abs=thr_nn,
            device=args.device,
        )
        metrics = evaluate_model(ecfg)
        results.append(
            {
                "method": name,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mle_mean": metrics["mle_mean"],
                "mle_std": metrics["mle_std"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
            }
        )

    df = pd.DataFrame(results)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSaved metrics to {out_path}")


if __name__ == "__main__":
    main()
