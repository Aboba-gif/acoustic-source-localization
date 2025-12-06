# acoustic-source-localization

### pipeline:

1) Generate a dataset:
```bash
python scripts/generate_dataset.py --config configs/simulator_default.yaml
```

2) Train models:
```bash
python scripts/train_unet_complex.py   --config configs/unet_complex.yaml
python scripts/train_shallow_cnn.py    --config configs/shallow_cnn.yaml
python scripts/train_unet_magnitude.py --config configs/unet_magnitude.yaml
```

3) Evaluate all methods and get a table:
```bash
python scripts/evaluate_all.py --config configs/eval.yaml
```

A complete pipeline was developed and tested in a [Google Colab notebook](https://colab.research.google.com/drive/1eDi8aRDEs-BNCxme_IZGul2nBNvr_nZg?usp=sharing) notebook, which can be used for experimentation and reproduction.