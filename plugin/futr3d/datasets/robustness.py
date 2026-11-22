"""Evaluation-only adapters; the clean NuScenesDataset is not modified."""
import os

import numpy as np

from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets.builder import DATASETS
from mmdet3d.datasets.pipelines.robustness import load_robustness_metadata


@DATASETS.register_module()
class NuScenesRobustnessDataset(NuScenesDataset):
    """Provide current GT for object failure, or toolkit frame/calibration data.

    Keep the checkpoint's native camera order: FUTR3D learns per-slot weights.
    Never mutate data_infos, tokens, GT, or the evaluation coordinate frame.
    """

    def __init__(self, corruption, noise_metadata, **kwargs):
        if not kwargs.get('test_mode', False):
            raise ValueError('NuScenesRobustnessDataset is evaluation-only')
        self.corruption = dict(corruption)
        kind = self.corruption.get('type')
        if kind not in ('object_failure', 'frame_stuck', 'calibration'):
            raise ValueError(f'Unsupported dataset corruption: {kind}')
        if kind == 'frame_stuck':
            if (self.corruption.get('sensor') not in ('lidar', 'camera')
                    or self.corruption.get('ratio') not in range(10, 100, 10)
                    or self.corruption.get('mode')
                    not in ('discrete', 'consecutive')):
                raise ValueError('Invalid frame-stuck sensor, ratio or mode')
        if kind == 'calibration' and self.corruption.get('mode') not in (
                'single', 'all'):
            raise ValueError('Calibration mode must be single or all')
        self.noise = load_robustness_metadata(noise_metadata)
        super().__init__(**kwargs)

    def get_data_info(self, index):
        results = super().get_data_info(index)
        kind = self.corruption['type']
        if kind == 'object_failure':
            # Test mode normally omits annotations; load current GT solely for
            # the reference foreground-removal transform, never for inference.
            results['ann_info'] = self.get_ann_info(index)
        elif kind == 'frame_stuck':
            sensor = self.corruption['sensor']
            ratio, mode = self.corruption['ratio'], self.corruption['mode']
            if sensor == 'lidar':
                path = results['pts_filename']
                flag = self.noise['lidar'][os.path.basename(
                    path)]['noise']['drop_frames'][ratio][mode]
                if flag['stuck'] and flag['replace']:
                    replacement = self.noise['lidar'][
                        flag['replace']]['mmdet_info']
                    results['pts_filename'] = os.path.join(
                        os.path.dirname(path), flag['replace'])
                    results['sweeps'] = replacement['sweeps']
                    results['timestamp'] = replacement['timestamp'] / 1e6
            else:
                paths = []
                for path in results['img_filename']:
                    flag = self.noise['camera'][os.path.basename(
                        path)]['noise']['drop_frames'][ratio][mode]
                    if flag['stuck'] and flag['replace']:
                        path = os.path.join(os.path.dirname(path),
                                            flag['replace'])
                    paths.append(path)
                results['img_filename'] = paths
        else:
            mode = self.corruption['mode']
            lidar2img, camera2lidar = [], []
            # Same ordering as the parent dataset, with current intrinsics.
            for cam in self.data_infos[index]['cams'].values():
                noise = self.noise['camera'][os.path.basename(
                    cam['data_path'])]['noise']['extrinsics_noise']
                rotation = noise[f'{mode}_noise_sensor2lidar_rotation']
                translation = noise[f'{mode}_noise_sensor2lidar_translation']
                inverse_r = np.linalg.inv(rotation)
                inverse_t = translation @ inverse_r.T
                lidar2cam = np.eye(4)
                lidar2cam[:3, :3] = inverse_r.T
                lidar2cam[3, :3] = -inverse_t
                intrinsic = np.eye(4)
                intrinsic[:3, :3] = cam['cam_intrinsic']
                lidar2img.append(intrinsic @ lidar2cam.T)
                camera2lidar_rt = np.eye(4, dtype=np.float32)
                camera2lidar_rt[:3, :3] = rotation
                camera2lidar_rt[:3, 3] = translation
                camera2lidar.append(camera2lidar_rt)
            results['lidar2img'] = lidar2img
            results['camera2lidar'] = camera2lidar
        return results
