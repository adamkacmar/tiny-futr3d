import math
import torch
from mmdet3d.datasets.builder import PIPELINES


@PIPELINES.register_module()
class RobustnessLiDARFOV:
    def __init__(self, angle_range=(-60, 60)):
        self.min_angle, self.max_angle = angle_range

    def __call__(self, results):
        points = results["points"]
        xyz = points.tensor[:, :3]

        # Match the nuScenes-R benchmark's angular convention.
        angles = torch.atan2(xyz[:, 0], xyz[:, 1])
        angles = angles * (180.0 / math.pi)

        keep = (
            (angles >= self.min_angle)
            & (angles <= self.max_angle)
        )

        results["points"] = points[keep]
        return results