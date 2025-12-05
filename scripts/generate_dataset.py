import argparse
from pathlib import Path

import h5py
import json
import numpy as np
import yaml
from tqdm import tqdm

from acoustic_loc.config import (
    RoomConfig,
    GridConfig,
    SourcesConfig,
    AcousticsConfig,
    DatasetConfig,
    FullSimConfig,
)
from acoustic_loc.simulator import AcousticSimulator


def load_config(path: str) -> FullSimConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    room = RoomConfig(**cfg_dict["room"])
    grid = GridConfig(**cfg_dict["grid"])
    sources = SourcesConfig(**cfg_dict["sources"])
    acoustics = AcousticsConfig(**cfg_dict["acoustics"])
    dataset = DatasetConfig(**cfg_dict["dataset"])
    return FullSimConfig(room=room, grid=grid, sources=sources, acoustics=acoustics, dataset=dataset)


def save_split_to_h5(h5_path: Path, samples: list[dict]) -> None:
    if not samples:
        print(f"Skipping {h5_path}, no samples to save.")
        return

    n_samples = len(samples)
    grid_shape = samples[0]["pressure_real"].shape  # (H, W)

    h5_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        d_real = f.create_dataset("pressure_real", (n_samples, *grid_shape), dtype="float32")
        d_imag = f.create_dataset("pressure_imag", (n_samples, *grid_shape), dtype="float32")
        d_maps = f.create_dataset("source_maps", (n_samples, *grid_shape), dtype="float32")
        d_ids = f.create_dataset("sample_ids", (n_samples,), dtype="int64")

        for i, s in enumerate(tqdm(samples, desc=f"Writing {h5_path.name}")):
            d_real[i, :, :] = s["pressure_real"]
            d_imag[i, :, :] = s["pressure_imag"]
            d_maps[i, :, :] = s["source_maps"]
            d_ids[i] = s["sample_id"]

    print(f"Saved {n_samples} samples to {h5_path}")


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sim = AcousticSimulator(cfg)

    # общее число сцен
    n_total = cfg.dataset.n_train + cfg.dataset.n_val + cfg.dataset.n_test

    h5_dir = Path(cfg.dataset.h5_out_dir)
    json_dir = Path(cfg.dataset.json_out_dir)
    h5_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {n_total} samples...")
    all_samples = []

    for i in tqdm(range(n_total), desc="Generating samples"):
        sample_id = i + 1
        p_complex, s_map, meta = sim.generate_scene()

        meta["sample_id"] = int(sample_id)

        # сохраняем JSON в формате sample_000001_info.json
        json_path = json_dir / f"sample_{sample_id:06d}_info.json"
        json_dir.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        all_samples.append(
            {
                "sample_id": sample_id,
                "pressure_real": np.real(p_complex).astype("float32"),
                "pressure_imag": np.imag(p_complex).astype("float32"),
                "source_maps": s_map.astype("float32"),
            }
        )

    print("Shuffling and splitting into train/val/test...")
    rng = np.random.default_rng()  # можно зафиксировать seed
    rng.shuffle(all_samples)

    n_train = cfg.dataset.n_train
    n_val = cfg.dataset.n_val
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train : n_train + n_val]
    test_samples = all_samples[n_train + n_val :]

    save_split_to_h5(h5_dir / "train.h5", train_samples)
    save_split_to_h5(h5_dir / "val.h5", val_samples)
    save_split_to_h5(h5_dir / "test.h5", test_samples)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to simulator YAML config.",
    )
    main(parser.parse_args())
