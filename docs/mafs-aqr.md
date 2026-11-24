# MAFS with adaptive query routing

## Integration

In each FUTR3D decoder layer, MAFS samples LiDAR BEV features with multi-scale
deformable attention and projects the query reference point into the cameras
to sample and aggregate image features. The two output projections produce
one LiDAR vector and one camera vector per query, each shaped `[B, Q, 256]`.
The original fusion MLP consumes their camera-first concatenation
`[B, Q, 512]`. Its output enters the cross-attention residual and the rest of
the decoder layer. [Attention implementation](../plugin/futr3d/models/utils/futr3d_attention.py)

MoME projects query reference points into LiDAR and camera feature maps and
uses the Local Attention Mask to restrict routing cross-attention to nearby
features. It assigns queries by argmax to LiDAR, camera, or fused expert
decoding. The paper describes pretraining the experts on clean inputs and
then freezing them while supervising the router with cross-entropy on
scene-level sensor-drop labels. The public code broadcasts each scene label
over its queries. Its `ModalMask3D` uses a 50% clean, 25% camera-drop,
25% LiDAR-drop split, whereas the paper describes thirds.
[MoME paper, §3](https://arxiv.org/html/2503.19776v1#S3) · [Router code](https://github.com/konyul/MoME/blob/main/projects/mmdet3d_plugin/models/utils/multi_expert.py) · [Sensor-drop code](https://github.com/konyul/MoME/blob/main/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py)

This implementation scores the **already sampled and projected** LiDAR and
camera vectors with an MLP at each layer. The router sees LiDAR channels
first. Its two weights multiply the corresponding feature vectors before
the original camera-first fusion. There is one shared decoder stack, with a
separate router in each layer. The router adds 198,162 parameters at the
default width across six layers.

## Routing and training

For a query with LiDAR and camera vectors $f_L$ and $f_C$, the router computes
$z = \operatorname{MLP}([f_L; f_C])$ and $p = \operatorname{softmax}(z)$.
The three classes are ordered LiDAR only, camera only, and both, with
corresponding masks $(1,0)$, $(0,1)$, and $(1,1)$.

- **`none`** uses the original fusion directly and allocates no router.
- **`hard`** chooses the mask $m$ of $\arg\max z$ for each query. During
  training, it applies $m + (w - \operatorname{stopgrad}(w))$ so the
  forward values are binary and the backward pass differentiates through
  the soft weights $w$. Inference uses deterministic argmax masks.
- **`soft`** applies $w = (p_L + p_{LC},\ p_C + p_{LC})$ in training and
  inference. Both routed modes pass $[w_C f_C; w_L f_L]$ to fusion, with
  the straight-through hard weights substituted for $w$ in hard mode.

Fine-tuning loads the multimodal checkpoint via `load_from`. The new routers
start from the model's ordinary initialization. The original detection loss
updates the routers, fusion layers, and decoder jointly. The VoVNet config
retains its existing backbone freeze policy. Unlike MoME's router training,
this loss requires no sensor-drop labels or auxiliary routing cross-entropy.
Hard and soft should each be fine-tuned in their intended mode.

The existing deterministic `RobustnessLiDARDrop` and `RobustnessCameraDrop`
transforms can also run in training pipelines. A future stochastic
augmentation can reuse them by applying LiDAR dropout after sweep
aggregation and camera dropout before normalization. To preserve zero raw
pixels, camera dropout should also follow photometric distortion during
training. This experiment inherits the original training pipeline and
uses the existing corruption configs at evaluation.
[Robustness transforms](../mmdet3d/datasets/pipelines/robustness.py) · [Evaluation suite](robustness-evaluation.md)

## Config and diagnostics

The base model defaults to `routing_mode='none'`. The
[`lidar_0075v_cam_vov_aqr.py`](../plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov_aqr.py)
config enables `hard` and inherits the clean validation pipeline and its
checkpoint path. All router options belong to the cross-attention entry
`model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs.1`.

| Option | Values |
| --- | --- |
| `routing_mode` | `none`, `hard`, `soft` |
| `routing_hidden_dims` | Positive hidden width, default `64` |
| `routing_diagnostics` | `False` by default, `True` captures the latest batch per layer |

With capture enabled, each layer's `attentions[1].query_router.last_routing`
holds detached logits, probabilities, and weights on the model device. The
three `route_percentages` are over all queries in that batch. In soft mode
they summarize the largest logit per query. Inspect `weights` for the actual
continuous factors. Snapshots are replaced on every forward call, so copy
them immediately when comparing matched clean and corrupted batches. For a
built detector called `model`, read the snapshots after forward and inspect
router gradients after `loss.backward()` and before zeroing them.

```python
for layer in model.pts_bbox_head.transformer.decoder.layers:
    router = layer.attentions[1].query_router
    print(router.last_routing['route_percentages'].cpu().tolist())
    grad = router.mlp[-1].weight.grad
    if grad is not None:
        print(grad.norm().item())
```

## Commands

From the repository root in the existing `futr3d` Conda environment.

```bash
conda activate futr3d
export PYTHONPATH="$PWD${PYTHONPATH:+:${PYTHONPATH}}"
python -B tools/verify_mafs_aqr.py
```

The CPU verifier loads the original checkpoint strictly in disabled mode
and checks that only router keys are missing when enabling routing. It
checks exact disabled-mode predictions, query-specific binary masks,
continuous weights, finite outputs with zeroed or missing feature banks,
and detection-loss gradients through all six routers and fusion layers.
Its per-layer failure diagnostics use synthetic feature pyramids.

For future fine-tuning runs, the same checkpoint, seed, and training
schedule apply to all three modes. The `none` run controls for additional
fine-tuning. Use fresh work directories.

```bash
cfg_dir=plugin/futr3d/configs/lidar_cam
config="$cfg_dir/lidar_0075v_cam_vov_aqr.py"
attention=model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs.1

for mode in none hard soft; do
    python -B tools/train.py "$config" --gpu-id 0 --seed 0 \
        --work-dir "work_dirs/mafs_aqr_$mode" \
        --cfg-options "$attention.routing_mode=$mode" || break
done
```

For each fine-tuned checkpoint, set `mode` to the training mode. To evaluate
the original pretrained baseline, set `mode=none`, use the checkpoint path
from the AQR config, and set `run=futr3d_pretrained` for distinct outputs.
Run the clean validation split first, then the same validation samples
under each chosen corruption.

```bash
mode=hard
run="mafs_aqr_$mode"
checkpoint="work_dirs/$run/latest.pth"
for scenario in clean lidar_drop_all camera_drop_all fov60; do
    output_dir="work_dirs/$run/eval_$scenario"
    mkdir -p "$output_dir"
    python -B tools/test.py "$cfg_dir/lidar_0075v_cam_vov_eval_$scenario.py" \
        "$checkpoint" --gpu-id 0 --seed 0 --eval bbox \
        --cfg-options "$attention.routing_mode=$mode" \
        --out "$output_dir/predictions.pkl" \
        --eval-options "jsonfile_prefix=$output_dir/results" || break
done
```

Record the checkpoint, resolved config, command, scenario, severity, and
validation mAP and NDS. Report the absolute drop from each run's clean score
and the relative drop `(clean - corrupted) / clean` when the clean score is
positive.

## Scope and files

The router can mask sampled channels and can accept one missing feature bank
at the detection head. The detector's raw sensor entry points still expect
the baseline input representations. A hard route can select a failed
modality because choices are learned from the sampled features. Both
available modalities are sampled before routing, so masking offers no
backbone or sampling savings. Real-data accuracy, robustness, and the
benefit of sensor-drop training remain to be measured.

- [`query_router.py`](../plugin/futr3d/models/utils/query_router.py) adds the MLP, masking, and snapshots.
- [`futr3d_attention.py`](../plugin/futr3d/models/utils/futr3d_attention.py) adds routing before fusion and imports the existing MMCV CPU sampling fallback.
- [The AQR config](../plugin/futr3d/configs/lidar_cam/lidar_0075v_cam_vov_aqr.py) enables routing and inherits clean validation.
- [`verify_mafs_aqr.py`](../tools/verify_mafs_aqr.py) runs the CPU verification.
