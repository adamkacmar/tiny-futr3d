# Evaluation only; one independent corruption of the working clean baseline.
_base_ = ['./lidar_0075v_cam_vov_eval_clean.py']

from plugin.futr3d.robustness_scenarios import dataset_overrides as _build

robustness_scenario = 'camera_drop_all'
data = _build(robustness_scenario)
del _build
