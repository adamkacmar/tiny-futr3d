#!/usr/bin/env python3
"""CPU-only paired sample/reference verification for the robustness manifest."""
import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import mmcv
import numpy as np
import torch
from mmcv import Config

import plugin.futr3d  # Registers the evaluation adapters and image transforms.
from plugin.futr3d.robustness_scenarios import load_manifest
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.core.points import BasePoints
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.robustness import load_robustness_metadata


def reference_classes(path):
    text = Path(path).read_text()
    classes = {}
    for block in re.findall(r'```python\s*\n(.*?)```', text, re.S):
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            node.decorator_list = []
            node.bases = [
            ]  # Instantiate without invoking the dataset constructor.
            module = ast.fix_missing_locations(
                ast.Module(body=[node], type_ignores=[]))
            namespace = dict(np=np,
                             mmcv=mmcv,
                             os=os,
                             BasePoints=BasePoints,
                             box_np_ops=box_np_ops)
            exec(compile(module, str(path), 'exec'), namespace)
            classes[node.name] = namespace[node.name]
    return classes


def reference_mome(path):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'ModalMask3D')
    node.decorator_list = []
    namespace = dict(np=np)
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=[node],
                                                 type_ignores=[])), str(path),
            'exec'), namespace)
    return namespace['ModalMask3D']


def same_annotations(left, right):
    for key in [
            'gt_boxes', 'gt_names', 'gt_velocity', 'num_lidar_pts',
            'num_radar_pts', 'valid_flag'
    ]:
        # equal_nan is not supported for string arrays in this NumPy version.
        if left[key].dtype.kind in 'fc':
            assert np.array_equal(left[key], right[key], equal_nan=True), key
        else:
            assert np.array_equal(left[key], right[key]), key


def choose_index(scenario, clean, noise):
    kind, p = scenario['corruption'], scenario['parameters']
    for i, info in enumerate(clean.data_infos):
        if len(info['sweeps']) < 9:
            continue
        lidar = noise['lidar'][os.path.basename(info['lidar_path'])]
        if kind == 'object_failure' and not lidar['noise']['object_failure']:
            continue
        if kind == 'frame_stuck':
            entries = ([lidar] if p['sensor'] == 'lidar' else [
                noise['camera'][os.path.basename(c['data_path'])]
                for c in info['cams'].values()
            ])
            flags = [
                e['noise']['drop_frames'][p['ratio']][p['mode']]
                for e in entries
            ]
            if not any(f['stuck'] and f['replace'] for f in flags):
                continue
        return i
    raise AssertionError('No affected sample with sweeps: ' + scenario['name'])


def reference_dataset_data(ref_class, dataset, scenario, noise, index):
    ref = ref_class.__new__(ref_class)
    ref.data_infos = dataset.data_infos
    ref.modality = dataset.modality
    ref.test_mode = True
    p = scenario['parameters']
    ref.drop_frames = scenario['corruption'] == 'frame_stuck'
    ref.extrinsics_noise = scenario['corruption'] == 'calibration'
    ref.noise_sensor_type = p.get('sensor', 'camera')
    ref.drop_ratio, ref.drop_type = p.get('ratio'), p.get('mode')
    ref.extrinsics_noise_type = p.get('mode')
    ref.noise_data = noise[ref.noise_sensor_type]
    return ref.get_data_info(index)


def reference_beam_cloud(info, reduce):
    points = reduce(
        np.fromfile(info['lidar_path'], dtype=np.float32).reshape(-1, 5), 4)
    points[:, 4] = 0
    clouds = [points]
    for sweep in info['sweeps'][:9]:
        pts = reduce(
            np.fromfile(sweep['data_path'], dtype=np.float32).reshape(-1, 5),
            4)
        pts = pts[~((np.abs(pts[:, 0]) < 1) & (np.abs(pts[:, 1]) < 1))]
        pts[:, :3] = pts[:, :3] @ sweep['sensor2lidar_rotation'].T
        pts[:, :3] += sweep['sensor2lidar_translation']
        pts[:, 4] = (info['timestamp'] - sweep['timestamp']) / 1e6
        # Match the loader's subtraction of two separately scaled timestamps.
        pts[:, 4] = info['timestamp'] / 1e6 - sweep['timestamp'] / 1e6
        clouds.append(pts)
    return torch.from_numpy(np.concatenate(clouds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',
                        choices=['paper', 'toolkit', 'all'],
                        default='all')
    parser.add_argument('--scenario', nargs='+')
    parser.add_argument(
        '--toolkit',
        type=Path,
        default=ROOT.parent /
        'lidar-camera-robust-benchmark/Toolkit_nuscenes_English.md')
    parser.add_argument(
        '--mome-transform',
        type=Path,
        default=ROOT.parent /
        'MoME/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py')
    parser.add_argument('--report', type=Path)
    parser.add_argument(
        '--inference',
        action='store_true',
        help='Also run one GPU inference per selected scenario')
    args = parser.parse_args()
    os.chdir(str(ROOT))
    torch.set_num_threads(2)
    refs = reference_classes(args.toolkit)
    mome = reference_mome(args.mome_transform)
    reference_path = ROOT / 'tests/robustness/references/bevfusion_loading_utils.py'
    spec = importlib.util.spec_from_file_location('beam_reference',
                                                  str(reference_path))
    beam_reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(beam_reference)
    manifest = load_manifest()
    rows = [
        s for s in manifest['scenarios']
        if args.suite == 'all' or s['suite'] == args.suite
    ]
    if args.scenario:
        unknown = set(args.scenario) - {s['name'] for s in rows}
        if unknown:
            parser.error('Unknown/unselected scenarios: ' +
                         ', '.join(sorted(unknown)))
        rows = [s for s in rows if s['name'] in args.scenario]
    metadata_path = os.environ.get('NUSCENES_R_METADATA',
                                   manifest['noise_metadata'])
    noise = load_robustness_metadata(metadata_path)
    clean_config = Config.fromfile(
        str(ROOT / manifest['scenarios'][0]['config']))
    clean = build_dataset(clean_config.data.test)
    clean_tokens = [i['token'] for i in clean.data_infos]
    assert len(clean_tokens) == 6019 and len(set(clean_tokens)) == 6019
    model = None
    if args.inference:
        from mmcv.parallel import MMDataParallel, collate
        from mmdet3d.models import build_model
        model_config = copy.deepcopy(clean_config.model)
        model_config.train_cfg = None
        model = build_model(model_config,
                            test_cfg=clean_config.get('test_cfg'))
        checkpoint = torch.load(str(ROOT / manifest['checkpoint']),
                                map_location='cpu')
        state = checkpoint.get('state_dict', checkpoint)
        state = {
            (k[7:] if k.startswith('module.') else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=True)
        model.CLASSES = clean.CLASSES
        model = MMDataParallel(model.cuda(), device_ids=[0])
        model.eval()
    report = []
    for scenario in rows:
        name, kind, params = scenario['name'], scenario[
            'corruption'], scenario['parameters']
        config = Config.fromfile(str(ROOT / scenario['config']))
        assert config.model == clean_config.model
        assert config.train_pipeline == clean_config.train_pipeline
        assert config.data.train == clean_config.data.train
        assert config.data.val.pipeline == config.data.test.pipeline
        dataset = build_dataset(config.data.test)
        assert [i['token'] for i in dataset.data_infos] == clean_tokens
        assert dataset.test_mode and dataset.data_root == clean.data_root
        index = choose_index(scenario, clean, noise)
        original = copy.deepcopy(clean.data_infos[index])
        same_annotations(dataset.data_infos[index], original)
        data, base = dataset.get_data_info(index), clean.get_data_info(index)
        assert data['sample_idx'] == base['sample_idx'] == original['token']
        names = [
            os.path.basename(p).split('__')[1] for p in data['img_filename']
        ]
        assert names == list(original['cams'])
        affected = kind == 'clean'
        if kind in ('frame_stuck', 'calibration'):
            expected = reference_dataset_data(refs['NuScenesDataset'], dataset,
                                              scenario, noise, index)
            assert data['pts_filename'] == expected['pts_filename']
            assert data['timestamp'] == expected['timestamp']
            assert [s['data_path'] for s in data['sweeps']
                    ] == [s['data_path'] for s in expected['sweeps']]
            expected_names = [
                os.path.basename(p).split('__')[1]
                for p in expected['img_filename']
            ]
            for j, cam_name in enumerate(names):
                k = expected_names.index(cam_name)
                assert data['img_filename'][j] == expected['img_filename'][k]
                assert np.array_equal(data['lidar2img'][j],
                                      expected['lidar2img'][k])
                if kind == 'calibration':
                    cam = original['cams'][cam_name]
                    rec = noise['camera'][os.path.basename(
                        cam['data_path'])]['noise']['extrinsics_noise']
                    rt = data['camera2lidar'][j]
                    assert np.array_equal(
                        rt[:3, :3],
                        rec[params['mode'] +
                            '_noise_sensor2lidar_rotation'].astype(np.float32))
                    assert np.array_equal(
                        rt[:3, 3],
                        rec[params['mode'] +
                            '_noise_sensor2lidar_translation'].astype(
                                np.float32))
            if kind == 'calibration':
                deltas = [
                    float(np.max(np.abs(a - b)))
                    for a, b in zip(data['lidar2img'], base['lidar2img'])
                ]
                affected = all(delta > 0 for delta in deltas)
            elif params['sensor'] == 'lidar':
                affected = data['pts_filename'] != base['pts_filename']
                assert all(
                    np.array_equal(a, b)
                    for a, b in zip(data['lidar2img'], base['lidar2img']))
            else:
                affected = data['img_filename'] != base['img_filename']
        else:
            assert all(
                np.array_equal(a, b)
                for a, b in zip(data['lidar2img'], base['lidar2img']))
        dataset.pre_pipeline(data)
        counts = {}
        for transform in dataset.pipeline.transforms:
            cls = type(transform).__name__
            if cls == 'RobustnessLiDARFOV':
                expected = refs['LoadPointsFromMultiSweeps'].__new__(
                    refs['LoadPointsFromMultiSweeps'])
                expected.point_cloud_angle_range = params['angle_range']
                before = data['points']
                ref_points = expected.filter_point_by_angle(before).tensor
            elif cls == 'RobustnessObjectFailure':
                before = data['points']
                counts['before_object_failure'] = len(before)
                reference = refs['Randomdropforeground'].__new__(
                    refs['Randomdropforeground'])
                reference.noise_lidar_data = noise['lidar']
                expected = reference(copy.copy(data))['points'].tensor
            elif cls == 'RobustnessLiDARDrop':
                before = data['points'].clone()
                expected = mome(mode='test', mask_modal='points')({
                    'points':
                    before
                })['points'].tensor
            elif cls == 'RobustnessCameraDrop':
                before_images = data['img']
                if len(params['cameras']) == 6:
                    expected_images = mome(mode='test', mask_modal='image')({
                        'img':
                        before_images
                    })['img']
                else:
                    reference = refs['LoadMultiViewImageFromFiles'](
                        to_float32=True, drop_camera=params['cameras'])
                    # Reference expects fixed toolkit order. Align filenames to
                    # that order, run it unchanged, then compare by camera name.
                    order = reference.cam_orders
                    filenames = [
                        data['img_filename'][names.index(n)] for n in order
                    ]
                    out = reference({'img_filename': filenames})['img']
                    expected_images = [
                        out[order.index(n)].astype(np.float32) for n in names
                    ]
            elif cls == 'RobustnessCameraOcclusion':
                before_images = data['img']
                ref_mask = refs['LoadMaskMultiViewImageFromFiles'].__new__(
                    refs['LoadMaskMultiViewImageFromFiles'])
                ref_mask.noise_camera_data = noise['camera']
                ref_mask.mask_file = os.environ.get('NUSCENES_R_MASK_DIR',
                                                    manifest['mask_dir'])
                ref_mask.img_scale = None
                ref_mask.color_type = 'unchanged'
                ref_mask.to_float32 = True
                expected_images = ref_mask(
                    {'img_filename': data['img_filename']})['img']
            data = transform(data)
            if cls in ('LoadPointsFromMultiSweeps',
                       'LoadPointsFromMultiSweeps4Beams'):
                counts['fused'] = len(data['points'])
                if cls.endswith('4Beams'):
                    expected_cloud = reference_beam_cloud(
                        original, beam_reference.reduce_LiDAR_beams)
                    assert torch.equal(data['points'].tensor, expected_cloud)
                    assert len(expected_cloud) < len(
                        np.fromfile(original['lidar_path'],
                                    dtype=np.float32)) // 5 * 10
                    affected = True
            if cls == 'RobustnessLiDARFOV':
                assert torch.equal(data['points'].tensor, ref_points)
                counts['after_corruption'] = len(data['points'])
                affected = len(data['points']) < len(before)
            elif cls in ('RobustnessObjectFailure', 'RobustnessLiDARDrop'):
                assert torch.equal(data['points'].tensor, expected)
                counts['after_corruption'] = len(data['points'])
                affected = (
                    len(data['points']) < counts['before_object_failure']
                    if cls == 'RobustnessObjectFailure' else
                    torch.count_nonzero(data['points'].tensor).item() == 0)
            elif cls in ('RobustnessCameraDrop', 'RobustnessCameraOcclusion'):
                assert all(
                    np.array_equal(a, b)
                    for a, b in zip(data['img'], expected_images))
                changed = [
                    not np.array_equal(a, b)
                    for a, b in zip(data['img'], before_images)
                ]
                assert any(changed)
                if cls == 'RobustnessCameraDrop':
                    assert changed == [n in params['cameras'] for n in names]
                else:
                    assert all(changed)
                affected = True
        assert affected, name + ' did not affect the selected sample'
        points = data['points'][0].data
        images = data['img'][0].data
        metas = data['img_metas'][0].data
        assert points.ndim == 2 and points.shape[
            1] == 5 and points.dtype == torch.float32
        assert tuple(images.shape) == (6, 3, 928,
                                       1600) and images.dtype == torch.float32
        assert torch.isfinite(points).all() and torch.isfinite(images).all()
        assert metas['sample_idx'] == original['token']
        same_annotations(dataset.data_infos[index], original)
        same_annotations(clean.data_infos[index], original)
        # Exactly one corruption family; beam needs both loader replacements.
        pipeline_types = [t['type'] for t in config.data.test.pipeline]
        markers = [t for t in pipeline_types if t.startswith('Robustness')]
        assert len(markers) == (1 if kind in ('fov', 'lidar_drop',
                                              'object_failure', 'camera_drop',
                                              'occlusion') else 0)
        assert sum(
            t.endswith('4Beams')
            for t in pipeline_types) == (2 if kind == 'beam_reduction' else 0)
        entry = dict(name=name,
                     status='passed',
                     sample_index=index,
                     token=original['token'],
                     points=int(points.shape[0]),
                     image_shape=list(images.shape),
                     camera_order=names,
                     **counts)
        if kind == 'calibration': entry['lidar2img_max_abs_deltas'] = deltas
        if model is not None:
            with torch.no_grad():
                predictions = model(return_loss=False,
                                    rescale=True,
                                    **collate([data], samples_per_gpu=1))
            assert len(predictions) == 1
            prediction = predictions[0]['pts_bbox']
            assert torch.isfinite(prediction['scores_3d']).all()
            assert torch.isfinite(prediction['boxes_3d'].tensor).all()
            entry['inference_boxes'] = len(prediction['scores_3d'])
            entry['inference_status'] = 'passed'
        report.append(entry)
        print(json.dumps(entry), flush=True)
        del dataset, data, base, images, points, config
        gc.collect()
    # Inactive schedules and invalid combinations must not corrupt clean data.
    from mmdet3d.datasets.pipelines.robustness import RobustnessObjectFailure
    from mmdet3d.core.points import LiDARPoints
    from plugin.futr3d.datasets.robustness import NuScenesRobustnessDataset
    inactive = next(i for i in clean.data_infos if not noise['lidar'][
        os.path.basename(i['lidar_path'])]['noise']['object_failure'])
    pts = LiDARPoints(torch.tensor([[1., 2., 3., 4., 5.]]), points_dim=5)
    out = RobustnessObjectFailure(metadata_path)({
        'pts_filename':
        inactive['lidar_path'],
        'points':
        pts
    })
    assert out['points'] is pts
    try:
        NuScenesRobustnessDataset(corruption={
            'type': 'frame_stuck',
            'ratio': 100,
            'mode': 'discrete',
            'sensor': 'lidar'
        },
                                  noise_metadata=metadata_path,
                                  test_mode=True)
    except ValueError:
        pass
    else:
        raise AssertionError('Unsupported severity was accepted')
    payload = dict(
        cpu_only=not args.inference,
        scenarios=report,
        reference_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.toolkit, args.mome_transform, reference_path]
        })
    if args.report:
        with args.report.open('x') as stream:
            json.dump(payload, stream, indent=2)
            stream.write('\n')
    print(
        'PASS: {} scenarios; reference equivalence and preserved model/training/GT/tokens'
        .format(len(report)))


if __name__ == '__main__':
    main()
