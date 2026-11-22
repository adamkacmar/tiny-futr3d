BEVFusion reference snapshot for independent smoke-test comparison.

Source: https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/datasets/pipelines/loading_utils.py

Commit: `326653dc06e0938edf1aae7d01efcd158ba83de5`.
The source file is unmodified; its upstream license is in BEVFUSION_LICENSE.
Only `reduce_LiDAR_beams(..., 4)` is executed by the tests.

nuScenes-R reference classes are extracted at test time from the local
`Toolkit_nuscenes_English.md`, not reimplemented as test expectations.
MoME's ModalMask3D is extracted from its local released transform_3d.py.
