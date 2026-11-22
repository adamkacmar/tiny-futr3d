"""Build evaluation dataset overrides from the single scenario manifest."""
import copy
import json
import os
from pathlib import Path

from mmcv import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / 'configs/robustness/scenarios.json'


def load_manifest():
    with MANIFEST.open() as stream:
        return json.load(stream)


def dataset_overrides(name):
    manifest = load_manifest()
    scenario = next((s for s in manifest['scenarios'] if s['name'] == name),
                    None)
    if scenario is None:
        raise ValueError(f'Unknown robustness scenario: {name}')
    kind, params = scenario['corruption'], scenario['parameters']
    clean = Config.fromfile(
        str(REPO_ROOT /
            'plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov_eval_clean.py'
            ))
    dataset = copy.deepcopy(dict(clean.data.test))
    pipeline = dataset['pipeline']
    metadata = os.environ.get('NUSCENES_R_METADATA',
                              manifest['noise_metadata'])
    mask_dir = os.environ.get('NUSCENES_R_MASK_DIR', manifest['mask_dir'])
    # Every scenario starts from clean, never another corrupted configuration.
    assert not any(t['type'].startswith('Robustness') for t in pipeline)
    sweep_index = next(i for i, t in enumerate(pipeline)
                       if t['type'] == 'LoadPointsFromMultiSweeps')
    if kind == 'beam_reduction':
        if params['beams'] != 4:
            raise ValueError(
                'Only the verified four-beam setting is supported')
        for t in pipeline:
            if t['type'] in ('LoadPointsFromFile',
                             'LoadPointsFromMultiSweeps'):
                t['type'] += '4Beams'
    elif kind == 'lidar_drop':
        pipeline.insert(sweep_index + 1, dict(type='RobustnessLiDARDrop'))
    elif kind == 'fov':
        pipeline.insert(sweep_index + 1,
                        dict(type='RobustnessLiDARFOV', **params))
    elif kind == 'object_failure':
        pipeline[sweep_index + 1:sweep_index + 1] = [
            dict(type='LoadAnnotations3D',
                 with_bbox_3d=True,
                 with_label_3d=True),
            dict(type='RobustnessObjectFailure', noise_metadata=metadata)
        ]
    elif kind in ('camera_drop', 'occlusion'):
        i = next(i for i, t in enumerate(pipeline)
                 if t['type'] == 'LoadMultiViewImageFromFiles')
        transform = (dict(type='RobustnessCameraDrop',
                          cameras=params['cameras']) if kind == 'camera_drop'
                     else dict(type='RobustnessCameraOcclusion',
                               noise_metadata=metadata,
                               mask_dir=mask_dir))
        pipeline.insert(i + 1, transform)
    elif kind not in ('frame_stuck', 'calibration', 'clean'):
        raise ValueError(f'Unsupported scenario kind: {kind}')
    if kind in ('object_failure', 'frame_stuck', 'calibration'):
        dataset.update(type='NuScenesRobustnessDataset',
                       noise_metadata=metadata,
                       corruption=dict(type=kind, **params))
    return dict(val=copy.deepcopy(dataset), test=dataset)
