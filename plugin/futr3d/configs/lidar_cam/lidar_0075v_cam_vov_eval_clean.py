# Evaluation only: full nuScenes validation split; training/model inherited.
_base_ = ['./lidar_0075v_cam_vov.py']

# Explicit plugin directory is required by tools/test.py.
plugin_dir = 'plugin/futr3d/'

# Use the same nearest nine sweeps in both evaluation configurations.
test_pipeline = [{
    'type': 'LoadMultiViewImageFromFiles',
    'to_float32': True
}, {
    'type': 'LoadPointsFromFile',
    'coord_type': 'LIDAR',
    'load_dim': 5,
    'use_dim': 5,
    'file_client_args': {
        'backend': 'disk'
    }
}, {
    'type': 'LoadPointsFromMultiSweeps',
    'sweeps_num': 9,
    'use_dim': [0, 1, 2, 3, 4],
    'file_client_args': {
        'backend': 'disk'
    },
    'pad_empty_sweeps': True,
    'remove_close': True,
    'test_mode': True
}, {
    'type': 'NormalizeMultiviewImage',
    'mean': [103.53, 116.28, 123.675],
    'std': [57.375, 57.12, 58.395],
    'to_rgb': False
}, {
    'type': 'PadMultiViewImage',
    'size_divisor': 32
}, {
    'type':
    'MultiScaleFlipAug3D',
    'img_scale': (1333, 800),
    'pts_scale_ratio':
    1,
    'flip':
    False,
    'transforms': [{
        'type': 'GlobalRotScaleTrans',
        'rot_range': [0, 0],
        'scale_ratio_range': [1.0, 1.0],
        'translation_std': [0, 0, 0]
    }, {
        'type': 'RandomFlip3D'
    }, {
        'type': 'PointsRangeFilter',
        'point_cloud_range': [-54, -54, -5.0, 54, 54, 3.0]
    }, {
        'type':
        'DefaultFormatBundle3D',
        'class_names': [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
            'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ],
        'with_label':
        False
    }, {
        'type': 'Collect3D',
        'keys': ['points', 'img']
    }]
}]

eval_pipeline = [{
    'type': 'LoadPointsFromFile',
    'coord_type': 'LIDAR',
    'load_dim': 5,
    'use_dim': 5,
    'file_client_args': {
        'backend': 'disk'
    }
}, {
    'type': 'LoadPointsFromMultiSweeps',
    'sweeps_num': 9,
    'use_dim': [0, 1, 2, 3, 4],
    'file_client_args': {
        'backend': 'disk'
    },
    'pad_empty_sweeps': True,
    'remove_close': True,
    'test_mode': True
}, {
    'type':
    'DefaultFormatBundle3D',
    'class_names': [
        'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
        'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
    ],
    'with_label':
    False
}, {
    'type': 'Collect3D',
    'keys': ['points']
}]

data = dict(val=dict(pipeline=test_pipeline, test_mode=True),
            test=dict(pipeline=test_pipeline, test_mode=True))
load_from = 'work_dirs/lidar_0075_cam_vov_900q/lidar_0075_cam_vov_900q.pth'
