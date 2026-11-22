# Tiny-FUTR3D robustness evaluation

This suite evaluates the existing LiDAR+camera checkpoint on the complete
6,019-sample nuScenes **validation** split. It does not train or modify a model,
regenerate annotations, or write to sensor data, noise metadata, or checkpoints.
Every scenario starts from the working clean evaluation configuration and enables
one corruption family. Model parameters, training configuration, GT, sample tokens,
class ordering, spatial range, and evaluation protocol are inherited unchanged.

## Primary experiment matrix

`configs/robustness/scenarios.json` is the scenario manifest. Suite `paper`
contains clean plus the six columns of MoME's main sensor-failure comparison.

| Name | Failure/setting | Reference algorithm | Integration point | Required assets |
| --- | --- | --- | --- | --- |
| `clean` | No corruption | Existing working clean config | Unchanged | nuScenes |
| `fov60` | LiDAR FOV [-60, 60] degrees | Toolkit angular formula and inclusive bounds | After sweep fusion | nuScenes |
| `beam4` | Four LiDAR beams | BEVFusion elevation bands [7,9,11,13], original strict inequalities, float32 recurrence and -0.012 offset | Raw keyframe and each raw sweep, **before** sweep transformations and feature selection | nuScenes |
| `lidar_drop_all` | Complete LiDAR failure | Released MoME `ModalMask3D(mode='test', mask_modal='points')`: multiply the complete fused tensor by zero, retain point count and five columns | After sweep fusion | nuScenes |
| `object_failure_05` | Fixed frame flags generated with probability 0.5 | Toolkit `Randomdropforeground`: `points_in_rbbox`, remove union of all supplied GT boxes only when the frame flag is true | After sweep fusion and `LoadAnnotations3D`, before spatial range filtering | Noise PKL `lidar[basename].noise.object_failure` |
| `camera_drop_all` | Six dropped views | Zero raw images, preserving six slots and matrices, before normalization; matches MoME image masking | After image loading | nuScenes |
| `camera_occlusion` | Predefined obstacle/mud masks | `rot90(mask)`, `mmcv.imresize(mask,(w,h))`, `alpha=(mask/255)^3`, `alpha*image+(1-alpha)*mask`, float32 output | After image loading, before normalization and padding | Noise PKL `camera[basename].noise.mask_noise.mask_id`, `mask_<id>.jpg` |

The two existing clean/FOV60 configuration files are preserved byte-for-byte.
The other five configs inherit the clean file and call the shared scenario
builder. Ground truth loaded for object failure is used only to implement the
benchmark corruption; `Collect3D` still passes only points and images to inference.
This is the toolkit's GT-dependent corruption, not a deployable sensor model.

## Supplementary toolkit scenarios

Suite `toolkit` has 45 independent entries. These are not additional columns in
the selected MoME main-table comparison.

| Names | Severities/modes | Metadata and exact behavior |
| --- | --- | --- |
| `fov90` | [-90,90] degrees | Same angular convention and fused-cloud placement as FOV60 |
| `lidar_stuck_<ratio>_<mode>` | Integer ratios 10,20,...,90; `discrete`/`consecutive` | `lidar[basename].noise.drop_frames[ratio][mode]`; if `stuck` and `replace` nonempty, replace LiDAR basename and take replacement `mmdet_info.sweeps` and `timestamp/1e6`. Current sample token, annotations, evaluation poses, images, and calibration remain current. |
| `camera_stuck_<ratio>_<mode>` | Same nine ratios and two modes | `camera[basename].noise.drop_frames[ratio][mode]`; replace each flagged image basename, retaining current intrinsics/extrinsics. An empty replacement leaves current input unchanged. |
| `calibration_single`, `calibration_all` | One supplied perturbation setting; independent cameras / shared per-frame perturbation | Use `<mode>_noise_sensor2lidar_rotation/translation` from `camera[basename].noise.extrinsics_noise`; rebuild both `lidar2img` and `camera2lidar` with current intrinsics. Do not regenerate perturbations. |
| `camera_drop_cam_<view>` | Each of the six named views, independently | Zero that view before normalization. The reference API permits arbitrary camera subsets; these six single-view presets plus the all-view paper preset are selected experiments, not an official fixed severity ladder. |

For calibration, the generator declares rotation range `(1,5)` and translation
component magnitudes `(0.005,0.010)` m, with random signs. Its rotation code uses
`angle/360*pi`, **not** the usual degrees-to-radians conversion. The saved matrices
are authoritative; these configs apply them unchanged. `single` and `all` do not
represent severity levels.

Frame stuck is stale-frame replacement, **not** the total-modality drop in the
paper suite. Consecutive schedules are provided in the PKL and are never resampled.
The actual severity keys are integers, despite string-like keys in documentation
examples. Basenames identify metadata entries. No 100% stale-frame schedule is
invented.

## Reference correspondence and limitations

Sources:

- Local toolkit: `/home/kacmara/lidar-camera-robust-benchmark/Toolkit_nuscenes_English.md`
  and `create_noise_data_nuscenes.py`.
- [nuScenes-R toolkit](https://github.com/anonymous-benchmark/lidar-camera-robust-benchmark/blob/main/Toolkit_nuscenes_English.md).
- [MoME paper, experimental setup and Appendix B.2](https://arxiv.org/html/2503.19776v1).
- Released local MoME `projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py`,
  `ModalMask3D`, and its two failure configs.
- [BEVFusion reference](https://github.com/mit-han-lab/bevfusion/blob/326653dc06e0938edf1aae7d01efcd158ba83de5/mmdet3d/datasets/pipelines/loading_utils.py).
  The unmodified reference snapshot and upstream Apache-2.0 license are under
  `tests/robustness/references/`. Only its four-beam function is used in tests.
  The production wrappers keep Tiny-FUTR3D's existing loaders, feature layout,
  nearest-nine-sweep selection, and timestamp handling.

Important reproduction qualifications:

1. The paper says LiDAR points are removed, but released MoME code zeros the
   tensor instead. `lidar_drop_all` follows released code. Empty point clouds
   are not silently substituted; zero points remain at the origin and enter the
   existing voxelizer. No model branch is disabled and no checkpoint parameter
   changes. This can differ from an empty cloud or zero backbone features.
2. The paper describes per-box object failure probability. The supplied toolkit
   has one flag per **frame** and removes points in all its supplied boxes.
   This suite follows that toolkit and PKL: 3,057 of 6,019 frames are flagged.
   The setting 0.5 describes the generator probability, not an exact half of
   objects or frames. It is not resampled at evaluation time.
3. Preserve the checkpoint's native camera order:
   `FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT` (all with `CAM_`).
   The toolkit example orders `FRONT_LEFT, FRONT, FRONT_RIGHT, BACK_RIGHT, BACK,
   BACK_LEFT`. FUTR3D has learned per-camera-slot attention weights, so reordering
   even images and matrices together would alter baseline behavior. Corruptions
   use names/basenames; reference checks align by name. This is a documented
   integration difference, not an unqualified bit-for-bit copy of the full
   toolkit dataset class. All transformations consumed by FUTR3D stay aligned.
4. Zero images are zero **before** normalization; normalized images generally
   contain negative mean/std values. They are not expected to be zero tensors at
   the model input. Occlusion uses the supplied masks, not synthetic 3D obstacles.
5. The working FOV implementation excludes an undefined XY-origin point; the
   toolkit's original assert instead fails on its NaN angle. The working file
   is preserved. Its real-sample output matches the reference.
6. Inference uses the same existing spatial range filter. No extra annotations,
   object classes, sweeps, camera augmentations or checkpoint are introduced.
   Only the verified four-beam setting is exposed; other beam settings require
   separate verification.
7. Config `eval_pipeline` inherited from clean is a visualization helper, not the
   test pipeline used for metrics. These configs and tests cover inference/
   evaluation; corrupted point-cloud visualization through `show` is not covered.
8. Comparable corruption definitions alone do not reproduce published MoME
   numbers: architecture, training, checkpoint, image resolution and temporal
   context can differ. Treat published results as references until locally
   reproduced. Do not change each model's clean preprocessing to force agreement.

## Environment and paths

Use `futr3d` for Tiny-FUTR3D; do not run it in `mome`.
Verified interpreter: `/home/kacmara/.conda/envs/futr3d/bin/python`, Python 3.8,
PyTorch 1.10.1+cu111, MMCV 1.6.0, MMDetection 2.24.0.

Defaults:

- Checkpoint: `work_dirs/lidar_0075_cam_vov_900q/lidar_0075_cam_vov_900q.pth`.
- Annotations: `data/nuscenes/nuscenes_infos_val.pkl`.
- Metadata: `/home/kacmara/MoME/data/nuscenes-r/nuscenes_infos_val_with_noise.pkl`.
- Masks: `/home/kacmara/MoME/data/nuscenes-r/Mud_Mask_selected/`.

To relocate assets, set `NUSCENES_R_METADATA` and/or `NUSCENES_R_MASK_DIR` before
loading configs, testing, or running evaluation. Sensor filenames inside
annotations and replacement sweep metadata must still resolve from the repo root.

## List and preview (no inference, no output directories)

```bash
cd /home/kacmara/tiny-futr3d
conda activate futr3d
python -B tools/robustness_eval.py --list
python -B tools/robustness_eval.py --suite paper --dry-run
python -B tools/robustness_eval.py --scenario beam4 object_failure_05 --dry-run
python -B tools/robustness_eval.py --all --dry-run
```

The default suite is `paper`. Listing shows all 52 scenarios. Unknown names,
missing checkpoints and missing dependencies produce explicit errors.

## Evaluation (launch only when ready)

These commands run full GPU validation. They are not run during implementation.

```bash
# One of the five new paper failures:
python -B tools/robustness_eval.py --scenario beam4

# The remaining paper failures, preserving previously completed clean/FOV runs:
python -B tools/robustness_eval.py --scenario \
  beam4 lidar_drop_all object_failure_05 camera_drop_all camera_occlusion

# Clean plus all six paper failures:
python -B tools/robustness_eval.py --suite paper

# Selected supplementary experiments:
python -B tools/robustness_eval.py --scenario \
  lidar_stuck_50_discrete camera_stuck_50_consecutive calibration_single

# All supplementary scenarios, or the entire 52-entry matrix:
python -B tools/robustness_eval.py --suite toolkit
python -B tools/robustness_eval.py --all
```

Use `--gpu-id` to override GPU 0, `--checkpoint PATH` to explicitly override the
checkpoint, or `--output-root PATH` to change the parent output directory.
Seed is fixed at 0. The runner uses its own Python interpreter, so environment
activation applies to the evaluation subprocess as well.

Each command expands to the existing `tools/test.py CONFIG CHECKPOINT --gpu-id 0
--seed 0 --eval bbox --out OUTPUT.pkl --eval-options jsonfile_prefix=PREFIX`
workflow. It does not retrain. A failed experiment stops the selected batch;
completed results are retained, and the failing log remains available.

## Output and experiment records

Every invocation allocates `work_dirs/robustness/run-<unique>/`; it never reuses a
previous run directory. Every experiment gets a separate subdirectory:

- `run.json`: checkpoint SHA256/path, package versions, interpreter, CUDA, GPU,
  seed, manifest hash, Git revision/status, metadata/mask hashes and selection.
- `<scenario>/scenario.json`: scenario parameters and exact subprocess command.
- `<scenario>/config.py`: resolved config snapshot.
- `<scenario>/evaluation.log`: complete test/evaluation stdout and stderr.
- `<scenario>/outputs.pkl`: predictions from the full validation split.
- `<scenario>/predictions/pts_bbox/results_nusc.json`: nuScenes-format predictions.
- `<scenario>/predictions/pts_bbox/metrics_summary.json` and `metrics_details.json`:
  official evaluation results.
- `<scenario>/result.json`: mAP, NDS, runtime, checkpoint and scenario parameters.
- `results.json`: machine-readable comparison of completed/failed experiments.

mAP/NDS are stored as fractions. When the invocation includes clean, the combined
record also contains absolute drops in percentage points and relative percentage
drops from that clean result. For batches excluding clean, use the separately
completed matching baseline; the runner does not guess which old result to use.

## Lightweight verification

```bash
# CPU only; config loading, dataset construction, affected sample pipelines,
# original-reference equivalence, image/point dtypes/shapes, GT/tokens and settings:
python -B tools/verify_robustness.py --suite all \
  --report /tmp/robustness-verification-new.json

# Optional: one inference sample per selected scenario (not full validation):
python -B tools/verify_robustness.py --suite paper \
  --scenario beam4 lidar_drop_all object_failure_05 camera_drop_all camera_occlusion \
  --inference --report /tmp/robustness-inference-new.json
```

Report paths must not already exist. Default verification is CPU only. Reference
classes are extracted from the actual local toolkit; MoME zeroing is extracted
from its local released transform; beam outputs are compared against the pinned
upstream snapshot, including per-sweep placement before coordinate transforms.
Metadata-driven tests deliberately select flagged frames with nonempty
replacements, not only unaffected inputs. Reference frame/calibration outputs
are compared by camera name to account for the documented ordering difference.
The test also verifies an inactive object-failure flag leaves points unchanged.

Recorded verification results are in `docs/robustness-verification.json`.
These are smoke/reference checks, not measured detection accuracy. Full corrupted
mAP/NDS results remain to be measured using the evaluation commands above.

### Recorded smoke-test outcome

- All **52 scenarios** passed CPU config/dataset/affected-pipeline/reference
  checks (seven paper entries and 45 supplementary entries).
- All **five new paper failures** passed strict checkpoint loading and one-sample
  GPU inference on GPU 0, with 300 finite-scored boxes per sample. This is not
  validation-set accuracy and does not establish robustness performance.
- Three runner tests passed with a fake evaluator: no execution/output writes in
  preview, unique directories and metric/degradation recording, failure logging
  and early termination. Four edge tests passed, including missing sweeps,
  empty replacement names, inactive flags, and evaluation-only guards.
- Clean/FOV60 config SHA256 hashes match their pre-implementation values.
- Paired primary sample: `cc18fde20db74d30825b0b60ec511b7b` (validation index 1).
  Clean fused cloud: **262,864** points, **252,608** after spatial filtering.
  Four-beam cloud: **36,851** fused, **36,344** after spatial filtering.
  FOV60: **50,449** after angular filtering, **47,896** after spatial filtering.
  Object failure: **262,161** after foreground removal (**703 removed**),
  **251,905** after spatial filtering.
  Total LiDAR drop retains **262,864 zero rows**, as released MoME code does.
  Camera corruption scenarios preserve the clean point cloud.
- Six image tensors retain final shape `(6, 3, 928, 1600)`, dtype float32.
  Every cloud retains five float32 features. Calibration tests verify all six
  `lidar2img` matrices change and `camera2lidar` matches the supplied extrinsics.
- No full validation, training, package installation, dataset modification,
  checkpoint modification, or MoME repository modification was performed.

Additional fast runner/edge checks:

```bash
python -B tests/robustness/test_runner.py
python -B tests/robustness/test_edges.py
```
