import argparse
import yaml

from acoustic_loc.train import TrainConfig, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    tcfg = TrainConfig(
        model_name="shallow_cnn",
        in_channels=cfg_dict["model"]["in_channels"],
        out_channels=cfg_dict["model"]["out_channels"],
        base_channels=cfg_dict["model"]["base_channels"],
        batch_size=cfg_dict["training"]["batch_size"],
        num_epochs=cfg_dict["training"]["num_epochs"],
        lr=cfg_dict["training"]["lr"],
        weight_decay=cfg_dict["training"]["weight_decay"],
        lr_patience=cfg_dict["training"]["lr_patience"],
        lr_factor=cfg_dict["training"]["lr_factor"],
        early_stop_patience=cfg_dict["training"]["early_stop_patience"],
        device=cfg_dict["training"]["device"],
        train_h5=cfg_dict["data"]["train_h5"],
        val_h5=cfg_dict["data"]["val_h5"],
        input_repr="complex",
        log_dir=cfg_dict["logging"]["log_dir"],
        save_best_only=cfg_dict["logging"]["save_best_only"],
    )

    train_model(tcfg)


if __name__ == "__main__":
    main()
