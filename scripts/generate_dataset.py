import argparse
from pathlib import Path

import h5py
import numpy as np
import yaml
from tqdm import trange

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


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    sim = AcousticSimulator(cfg)

    out_dir = Path(cfg.dataset.h5_out_dir)
    json_dir = Path(cfg.dataset.json_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    splits = {
        "train": cfg.dataset.n_train,
        "val": cfg.dataset.n_val,
        "test": cfg.dataset.n_test,
    }

    for split, n_samples in splits.items():
        h5_path = out_dir / f"{split}.h5"
        with h5py.File(h5_path, "w") as h5:
            pr_real_ds = h5.create_dataset(
                "pressure_real", shape=(n_samples, cfg.grid.ny, cfg.grid.nx), dtype="f4"
            )
            pr_imag_ds = h5.create_dataset(
                "pressure_imag", shape=(n_samples, cfg.grid.ny, cfg.grid.nx), dtype="f4"
            )
            smap_ds = h5.create_dataset(
                "source_maps", shape=(n_samples, cfg.grid.ny, cfg.grid.nx), dtype="f4"
            )
            ids_ds = h5.create_dataset("sample_ids", shape=(n_samples,), dtype="i8")

            for idx in trange(n_samples, desc=f"Generating {split}"):
                p, s_map, meta = sim.generate_scene()
                pr_real_ds[idx] = np.real(p).astype("f4")
                pr_imag_ds[idx] = np.imag(p).astype("f4")
                smap_ds[idx] = s_map
                ids_ds[idx] = idx

                json_path = json_dir / f"{split}_{idx:06d}.json"
                sim.save_scene(p, s_map, meta, json_path)

        print(f"Saved {split} to {h5_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to simulator YAML config.")
    main(parser.parse_args())
