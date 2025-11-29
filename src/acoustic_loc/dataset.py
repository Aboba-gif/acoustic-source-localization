from pathlib import Path
from typing import Literal, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class AcousticH5Dataset(Dataset):
    """
    Dataset поверх HDF5:
      - pressure_real: (N, H, W)
      - pressure_imag: (N, H, W)
      - source_maps:   (N, H, W)
    """

    def __init__(
        self,
        h5_path: str | Path,
        input_repr: Literal["complex", "magnitude"] = "complex",
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.input_repr = input_repr

        self._h5 = h5py.File(self.h5_path, "r")
        self.pr_real = self._h5["pressure_real"]
        self.pr_imag = self._h5["pressure_imag"]
        self.smap = self._h5["source_maps"]

    def __len__(self) -> int:
        return self.pr_real.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_real = self.pr_real[idx].astype(np.float32)
        p_imag = self.pr_imag[idx].astype(np.float32)
        s_map = self.smap[idx].astype(np.float32)

        if self.input_repr == "complex":
            x = np.stack([p_real, p_imag], axis=0)  # (2, H, W)
        elif self.input_repr == "magnitude":
            mag = np.sqrt(p_real**2 + p_imag**2)
            x = mag[None, ...]  # (1, H, W)
        else:
            raise ValueError(f"Unknown input_repr: {self.input_repr}")

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(s_map[None, ...])  # (1, H, W)
        return x_tensor, y_tensor

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None
