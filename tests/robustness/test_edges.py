"""Small edge cases not covered by the affected-sample manifest smoke tests."""
import os
from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from mmcv import Config

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import plugin.futr3d
from plugin.futr3d.robustness_scenarios import load_manifest
from plugin.futr3d.datasets.robustness import NuScenesRobustnessDataset
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.robustness import (load_robustness_metadata,
                                                   reduce_lidar_to_four_beams)


class EdgeTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        os.chdir(str(ROOT))
        cls.manifest = load_manifest()
        cls.noise_path = os.environ.get('NUSCENES_R_METADATA',
                                        cls.manifest['noise_metadata'])
        cls.noise = load_robustness_metadata(cls.noise_path)
        cls.config = Config.fromfile(
            str(ROOT / cls.manifest['scenarios'][0]['config']))
        cls.clean = build_dataset(cls.config.data.test)

    def test_no_previous_frame_and_inactive_flags_leave_current_data(self):
        for sensor in ['lidar', 'camera']:
            for mode in ['discrete', 'consecutive']:
                config = Config.fromfile(
                    str(ROOT / (
                        'plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov_eval_'
                        + sensor + '_stuck_90_' + mode + '.py')))
                dataset = build_dataset(config.data.test)
                found = set()
                for i, info in enumerate(self.clean.data_infos):
                    entries = ([
                        self.noise['lidar'][os.path.basename(
                            info['lidar_path'])]
                    ] if sensor == 'lidar' else [
                        self.noise['camera'][os.path.basename(c['data_path'])]
                        for c in info['cams'].values()
                    ])
                    flags = [
                        e['noise']['drop_frames'][90][mode] for e in entries
                    ]
                    checks = [('no_previous',
                               all(not f['replace'] for f in flags)),
                              ('inactive', all(not f['stuck'] for f in flags))]
                    for label, selected in checks:
                        if not selected or label in found:
                            continue
                        actual, clean = dataset.get_data_info(
                            i), self.clean.get_data_info(i)
                        for key in [
                                'pts_filename', 'timestamp', 'sample_idx',
                                'img_filename'
                        ]:
                            self.assertEqual(actual[key], clean[key])
                        found.add(label)
                    if len(found) == 2:
                        break
                # At ratio 90, every camera might not be simultaneously inactive;
                # no-previous scene beginnings are always tested for both sensors.
                self.assertIn('no_previous', found)
                if sensor == 'lidar': self.assertIn('inactive', found)

    def test_beam_empty_cloud_and_zero_origin(self):
        self.assertEqual(
            reduce_lidar_to_four_beams(np.empty(
                (0, 5), dtype=np.float32)).shape, (0, 5))
        self.assertEqual(
            reduce_lidar_to_four_beams(np.zeros(
                (1, 5), dtype=np.float32)).shape, (0, 5))

    def test_beam_scene_beginning_pads_only_reduced_points(self):
        config = Config.fromfile(
            str(ROOT /
                'plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov_eval_beam4.py'
                ))
        dataset = build_dataset(config.data.test)
        i = next(i for i, info in enumerate(dataset.data_infos)
                 if not info['sweeps'])
        data = dataset.get_data_info(i)
        dataset.pre_pipeline(data)
        for t in dataset.pipeline.transforms:
            if type(t).__name__ == 'NormalizeMultiviewImage': break
            data = t(data)
        raw = np.fromfile(dataset.data_infos[i]['lidar_path'],
                          dtype=np.float32).reshape(-1, 5)
        keyframe = reduce_lidar_to_four_beams(raw)
        keyframe[:, 4] = 0
        far = keyframe[~((np.abs(keyframe[:, 0]) < 1) &
                         (np.abs(keyframe[:, 1]) < 1))]
        expected = np.concatenate([keyframe] + [far] * 9)
        self.assertTrue(
            torch.equal(data['points'].tensor, torch.from_numpy(expected)))

    def test_training_and_combined_corruptions_rejected(self):
        with self.assertRaises(ValueError):
            NuScenesRobustnessDataset(corruption={'type': 'object_failure'},
                                      noise_metadata=self.noise_path,
                                      test_mode=False)
        with self.assertRaises(ValueError):
            NuScenesRobustnessDataset(
                corruption={'type': ['object_failure', 'calibration']},
                noise_metadata=self.noise_path,
                test_mode=True)


if __name__ == '__main__':
    unittest.main()
