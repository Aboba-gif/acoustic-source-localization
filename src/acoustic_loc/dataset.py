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
      - sample_ids:    (N,)      [опционально]

    Полностью совместим с форматом, который создаёт scripts/generate_dataset.py.
    """

    def __init__(
        self,
        h5_path: str | Path,
        input_repr: Literal["complex", "magnitude"] = "complex",
    ):
        super().__init__()
        self.h5_path = Path(h5_path)
        self.input_repr = input_repr

        # Открываем HDF5 один раз и держим дескриптор открытым.
        # В конце экспериментов желательно вызвать .close().
        self._h5 = h5py.File(self.h5_path, "r")
        self.pr_real = self._h5["pressure_real"]
        self.pr_imag = self._h5["pressure_imag"]
        self.smap = self._h5["source_maps"]

        # sample_ids могут отсутствовать (например, в старых файлах)
        self._sample_ids = self._h5["sample_ids"] if "sample_ids" in self._h5 else None

    def __len__(self) -> int:
        return self.pr_real.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Возвращает:
          - x:  Tensor, (C, H, W)
          - y:  Tensor, (1, H, W)
          - sample_id: int (совпадает с JSON: sample_{id:06d}_info.json)
        """
        p_real = self.pr_real[idx].astype(np.float32)
        p_imag = self.pr_imag[idx].astype(np.float32)
        s_map = self.smap[idx].astype(np.float32)

        if self.input_repr == "complex":
            x = np.stack([p_real, p_imag], axis=0)  # (2, H, W)
        elif self.input_repr == "magnitude":
            mag = np.sqrt(p_real**2 + p_imag**2)
            x = mag[None, ...]  # (1, H, W)
        else:
            raise ValueError(f"Unknown input_repr: {self.input_repr!r}")

        if self._sample_ids is not None:
            sample_id = int(self._sample_ids[idx])
        else:
            # Fallback: если sample_ids нет, считаем id = idx+1,
            # что совместимо с generate_dataset.py
            sample_id = idx + 1

        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(s_map[None, ...])  # (1, H, W)
        return x_tensor, y_tensor, sample_id

    @property
    def sample_ids(self) -> np.ndarray | None:
        if self._sample_ids is None:
            return None
        return np.array(self._sample_ids[:], dtype=np.int64)

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None:
            self._h5.close()
            self._h5 = None

    def __del__(self) -> None:
        # на случай, если забыли явно закрыть
        try:
            self.close()
        except Exception:
            pass
