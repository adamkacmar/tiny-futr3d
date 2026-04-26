import os
import sys
import traceback

from mmcv import Config


repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)

import plugin.futr3d  # noqa: F401 - registers custom FUTR3D modules


CFG_LIST = [
    "plugin/futr3d/configs/lidar_only/lidar_0075v_300q.py",
    "plugin/futr3d/configs/lidar_only/lidar_0075v_900q.py",
    "plugin/futr3d/configs/lidar_only/lidar_0075v_900q_4b.py",
    "plugin/futr3d/configs/lidar_only/lidar_0075v_900q_1b.py",
    "plugin/futr3d/configs/cam_radar/cam_res101_radar.py",
    "configs/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d.py",
]


def build_from_cfg(cfg_path: str):
    cfg = Config.fromfile(cfg_path)

    # Avoid loading external pretrained weights when just counting params.
    if "model" in cfg and isinstance(cfg.model, dict):
        cfg.model.pop("pretrained", None)
        cfg.model.pop("init_cfg", None)

    try:
        from mmdet3d.models import build_model

        model = build_model(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
    except Exception:
        from mmdet3d.models import build_detector

        model = build_detector(
            cfg.model,
            train_cfg=cfg.get("train_cfg"),
            test_cfg=cfg.get("test_cfg"),
        )
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


for cfg_path in CFG_LIST:
    print("=" * 120)
    print("Config:", cfg_path)
    try:
        model = build_from_cfg(cfg_path)
        total, trainable = count_params(model)
        print("Total params     :", f"{total:,}")
        print("Trainable params :", f"{trainable:,}")
        print("Trainable ratio  :", f"{(trainable / total) * 100:.2f}%")
    except Exception as exc:
        print("FAILED:", str(exc))
        traceback.print_exc()
