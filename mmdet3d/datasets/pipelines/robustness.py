import os
from functools import lru_cache

import mmcv
import numpy as np
import torch
from mmdet3d.datasets.builder import PIPELINES
from mmdet3d.core.bbox import box_np_ops
from .loading import LoadPointsFromFile, LoadPointsFromMultiSweeps


@PIPELINES.register_module()
class RobustnessLiDARFOV:
    """Keep an inclusive sector in the fused keyframe LiDAR coordinates.

    nuScenes-R measures degrees from +Y towards +X,
    after transforming and concatenating sweeps. All features are retained.
    """

    def __init__(self, angle_range=(-60, 60)):
        if len(angle_range) != 2:
            raise ValueError('angle_range must contain two angles')
        self.min_angle, self.max_angle = angle_range
        if not -180 <= self.min_angle <= self.max_angle <= 180:
            raise ValueError(
                'angle_range must satisfy -180 <= min <= max <= 180')

    def __call__(self, results):
        points = results["points"]
        xy = points.tensor[:, :2].detach().cpu().numpy()

        # Use the toolkit formula verbatim, including its boundary rounding
        # and exclusion of the undefined XY origin (NaN).
        with np.errstate(divide='ignore', invalid='ignore'):
            angles = (np.arctan(xy[:, 0] / xy[:, 1]) +
                      (xy[:, 1] < 0) * np.pi + np.pi * 2) % (np.pi * 2)
        angles[angles > np.pi] -= np.pi * 2
        angles = angles / np.pi * 180
        keep = torch.as_tensor(
            (angles >= self.min_angle) & (angles <= self.max_angle),
            device=points.tensor.device)

        results["points"] = points[keep]
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'angle_range=({self.min_angle}, {self.max_angle}))')


@lru_cache(maxsize=1)
def load_robustness_metadata(filename):
    """Share read-only benchmark metadata across transforms in one process."""
    return mmcv.load(os.path.abspath(os.path.expanduser(filename)))


def reduce_lidar_to_four_beams(points):
    """BEVFusion reduce_LiDAR_beams(..., 4), in the raw sensor frame.

    Reference: mit-han-lab/bevfusion, datasets/pipelines/loading_utils.py.
    Preserve its float32 recurrence, strict bounds and chosen bands exactly.
    """
    if not points.flags.writeable:
        points = points.copy()
    pts = torch.from_numpy(points)
    radius = torch.sqrt(pts[:, 0].pow(2) + pts[:, 1].pow(2) + pts[:, 2].pow(2))
    theta = torch.asin(pts[:, 2] / radius)
    beam_range = torch.zeros(32)
    beam_range[0], beam_range[31] = 0.1862, -0.5353
    for i in range(1, 31):
        beam_range[i] = beam_range[i - 1] - 0.023275
    mask = torch.zeros(len(pts))
    for band in [7, 9, 11, 13]:
        mask = mask + ((theta < (beam_range[band - 1] - 0.012)) *
                       (theta > (beam_range[band] - 0.012)))
    return pts[mask.bool()].numpy()


@PIPELINES.register_module()
class LoadPointsFromFile4Beams(LoadPointsFromFile):
    """Reduce the keyframe before the existing loader selects features."""

    def _load_points(self, filename):
        points = super()._load_points(filename).reshape(-1, self.load_dim)
        return reduce_lidar_to_four_beams(points).reshape(-1)


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps4Beams(LoadPointsFromMultiSweeps):
    """Reduce each raw sweep before its sensor-to-keyframe transformation."""

    def _load_points(self, filename):
        points = super()._load_points(filename).reshape(-1, self.load_dim)
        return reduce_lidar_to_four_beams(points).reshape(-1)


@PIPELINES.register_module()
class RobustnessLiDARDrop:
    """MoME ModalMask3D test/points: zero the fused tensor, retaining shape.

    This is intentionally distinct from an empty cloud or stale-frame input.
    """

    def __call__(self, results):
        results['points'].tensor = results['points'].tensor * 0.0
        return results

    def __repr__(self):
        return 'RobustnessLiDARDrop(representation=zero_tensor)'


@PIPELINES.register_module()
class RobustnessObjectFailure:
    """nuScenes-R Randomdropforeground: one supplied flag per keyframe."""

    def __init__(self, noise_metadata):
        self.noise_lidar_data = load_robustness_metadata(
            noise_metadata)['lidar']

    def __call__(self, results):
        name = os.path.basename(results['pts_filename'])
        if self.noise_lidar_data[name]['noise']['object_failure']:
            boxes = results['gt_bboxes_3d'].tensor.numpy()
            points = results['points']
            masks = box_np_ops.points_in_rbbox(points.coord.numpy(), boxes)
            results['points'] = points[np.logical_not(masks.any(-1))]
        return results

    def __repr__(self):
        return 'RobustnessObjectFailure(schedule=provided_frame_flags)'


@PIPELINES.register_module()
class RobustnessCameraDrop:
    """Zero named raw images before normalization; retain camera slots."""

    def __init__(self, cameras):
        self.cameras = tuple(cameras)
        valid = {
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK',
            'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        }
        if not self.cameras or not set(self.cameras) <= valid:
            raise ValueError(
                'cameras must be a nonempty list of nuScenes views')

    def __call__(self, results):
        # Basename contains the camera name; never assume a view index/order.
        names = [
            os.path.basename(p).split('__')[1] for p in results['img_filename']
        ]
        if not set(self.cameras) <= set(names):
            raise ValueError('Requested dropped camera is absent from sample')
        results['img'] = [
            np.zeros_like(img) if name in self.cameras else img
            for name, img in zip(names, results['img'])
        ]
        return results

    def __repr__(self):
        return f'RobustnessCameraDrop(cameras={self.cameras})'


@PIPELINES.register_module()
class RobustnessCameraOcclusion:
    """Toolkit LoadMaskMultiViewImageFromFiles blending, before normalization."""

    def __init__(self, noise_metadata, mask_dir):
        self.noise_camera_data = load_robustness_metadata(
            noise_metadata)['camera']
        self.mask_dir = os.path.expanduser(mask_dir)

    @staticmethod
    def put_mask_on_img(img, mask):
        h, w = img.shape[:2]
        mask = np.rot90(mask)
        mask = mmcv.imresize(mask, (w, h), return_scale=False)
        alpha = np.power(mask / 255, 3)
        return alpha * img + (1 - alpha) * mask

    def __call__(self, results):
        images = []
        for filename, img in zip(results['img_filename'], results['img']):
            record = self.noise_camera_data[os.path.basename(filename)]
            mask_id = record['noise']['mask_noise']['mask_id']
            path = os.path.join(self.mask_dir, f'mask_{mask_id}.jpg')
            mask = mmcv.imread(path, 'unchanged')
            if mask is None:
                raise FileNotFoundError(path)
            # Toolkit casts its stacked result to float32 when to_float32=True.
            images.append(self.put_mask_on_img(img, mask).astype(np.float32))
        results['img'] = images
        return results

    def __repr__(self):
        return f'RobustnessCameraOcclusion(mask_dir={self.mask_dir!r})'
