# Project: Tiny-FUTR3D

## Research Context

This repository is part of a master's thesis investigating robust multimodal 3D object detection for autonomous driving.

The research focuses on:
- FUTR3D and its modality-agnostic object queries.
- MoME and its Adaptive Query Routing (AQR).
- LiDAR-camera sensor fusion.
- Robustness under sensor degradation and failure.
- Evaluation using nuScenes and nuScenes-R.

The objective is to compare detection accuracy, robustness, and potentially computational efficiency across different architectures and sensor configurations.

## Repository Structure

- `~/tiny-futr3d/` — Modified FUTR3D implementation (this repository).
- `~/MoME/` — MoME implementation.
- `~/lidar-camera-robust-benchmark/` — nuScenes-R benchmark toolkit.
- `~/MoME/data/nuscenes/` — nuScenes dataset.
- `~/MoME/data/nuscenes-r/` — nuScenes-R corruption metadata and masks.

Important files:
- `mmdet3d/datasets/nuscenes_dataset.py` — nuScenes dataset implementation.
- `mmdet3d/datasets/pipelines/loading.py` — Sensor data loading.
- `mmdet3d/datasets/pipelines/robustness.py` — Custom robustness transforms.
- `plugin/futr3d/configs/` — FUTR3D model configurations.

## Development Environment

- Development occurs on a remote Linux machine (`beast3`) through VS Code Remote SSH.
- Python environments are managed using Conda.
- The codebase uses PyTorch, MMCV and MMDetection3D.
- This repository contains customized dependencies and implementations. Do not assume compatibility with current upstream versions.
- Inspect the active environment before running commands or installing dependencies.

## Implementation Guidelines

1. Read relevant existing code before modifying it.
2. Prefer minimal, targeted changes over large refactoring.
3. Preserve existing model architectures and checkpoint compatibility.
4. Do not modify training configurations when implementing evaluation-only corruptions.
5. Implement corruption transforms as reusable components where practical.
6. Follow the actual nuScenes-R toolkit documentation for corruption semantics.
7. Preserve nuScenes sample tokens, ground truth and evaluation protocols.
8. Never overwrite datasets, annotations or checkpoints.
9. Do not initiate expensive GPU experiments without explicit approval.
10. Do not install, upgrade or downgrade dependencies without approval.

## Evaluation Requirements

- Always establish clean nuScenes baseline performance.
- Use identical validation samples and corruption severity when comparing models.
- Evaluate using nuScenes metrics, including mAP and NDS.
- Record the checkpoint, configuration, corruption settings and evaluation commands.
- Keep clean and corrupted evaluation configurations separate.
- Distinguish validation-set results from test-set results.
- Report both absolute performance and degradation relative to clean performance.

## Current Work

The immediate focus is integrating nuScenes-R robustness evaluation into Tiny-FUTR3D.

The first experiment is limited LiDAR field of view, initially using the angular range [-60°, 60°].

Existing changes include:
- A custom `RobustnessLiDARFOV` transform in `mmdet3d/datasets/pipelines/robustness.py`.
- Changes to `plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov.py`.

These changes require verification before running full evaluation.

Future experiments include LiDAR frame failures, object failures, camera frame failures, occlusion and calibration noise.

## Working Style

- Inspect repository code rather than guessing implementation details.
- Explain proposed changes briefly before substantial modifications.
- Prefer incremental implementation and verification.
- Provide executable commands with correct paths.
- Clearly distinguish verified results from assumptions.
- At the end of each task, summarize changed files, verification results and remaining issues.

Before making changes, explain the proposed modifications and wait for my approval. Never modify files without explicit confirmation.