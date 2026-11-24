#!/usr/bin/env python3
"""Check MAFS+AQR on CPU using synthetic feature pyramids and a baseline checkpoint."""
import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from mmcv import Config
from mmcv.utils import get_logger

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import plugin.futr3d  # Registers the model and attention.
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.models import build_model
from plugin.futr3d.models.utils.query_router import QueryRouter

CONFIG_DIR = ROOT / 'plugin/futr3d/configs/lidar_cam'
BASELINE = CONFIG_DIR / 'lidar_0075v_cam_vov.py'
AQR = CONFIG_DIR / 'lidar_0075v_cam_vov_aqr.py'
ATTENTION = 'model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs.1'
NUM_TEST_QUERIES = 12


def verify_router():
    lidar = torch.tensor([[[2., 4.], [0., 4.], [1., 4.]]])
    camera = torch.tensor([[[0., 5.], [2., 5.], [1., 5.]]])
    expected_masks = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]])
    hard = QueryRouter(2, 'hard', hidden_dims=2, diagnostics=True)
    with torch.no_grad():
        hard.mlp[0].weight.copy_(torch.tensor([[1., 0., 0., 0.],
                                              [0., 0., 1., 0.]]))
        hard.mlp[0].bias.zero_()
        hard.mlp[2].weight.copy_(torch.tensor([[1., 0.], [0., 1.], [0., 0.]]))
        hard.mlp[2].bias.copy_(torch.tensor([0., 0., 1.5]))
    for training in (True, False):
        hard.train(training)
        masked_lidar, masked_camera = hard(lidar, camera)
        assert torch.equal(hard.last_routing['weights'], expected_masks)
        assert torch.equal(masked_lidar, lidar * expected_masks[..., :1])
        assert torch.equal(masked_camera, camera * expected_masks[..., 1:])

    soft = QueryRouter(2, 'soft', hidden_dims=2, diagnostics=True)
    soft.load_state_dict(hard.state_dict(), strict=True)
    weighted_lidar, weighted_camera = soft(lidar, camera)
    probabilities = soft.last_routing['probabilities']
    weights = soft.last_routing['weights']
    torch.testing.assert_close(probabilities, hard.last_routing['probabilities'])
    lidar_only, camera_only, both = probabilities.unbind(dim=-1)
    torch.testing.assert_close(probabilities.sum(-1), torch.ones_like(both))
    assert ((weights > 0) & (weights < 1)).all()
    torch.testing.assert_close(weights[..., 0], lidar_only + both)
    torch.testing.assert_close(weights[..., 1], camera_only + both)
    torch.testing.assert_close(weighted_lidar, lidar * weights[..., :1])
    torch.testing.assert_close(weighted_camera, camera * weights[..., 1:])

    quiet = QueryRouter(2, 'hard').eval()
    quiet(lidar, camera)
    assert quiet.last_routing is None
    print('Router: per-query binary masks, soft weights and opt-in diagnostics passed.')


def build_from_checkpoint(cfg, state, mode):
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                        test_cfg=cfg.get('test_cfg'))
    if mode == 'none':
        model.load_state_dict(state, strict=True)
    else:
        # Initialize as tools/train.py does before loading the baseline weights.
        model.init_weights()
        loaded = model.load_state_dict(state, strict=False)
        router_keys = {key for key in model.state_dict() if '.query_router.' in key}
        assert router_keys and set(loaded.missing_keys) == router_keys, loaded
        assert not loaded.unexpected_keys, loaded
    return model


def synthetic_inputs(cross_attention):
    batch_size = 2
    channels = cross_attention.embed_dims
    num_cams = cross_attention.num_cams
    sizes = (8, 4, 2, 1)
    lidar = [torch.randn(batch_size, channels, size, size) for size in sizes]
    camera = [
        torch.randn(batch_size, num_cams, channels, size, size) for size in sizes
    ]
    # Map the entire model XY range into each synthetic camera's image.
    projection = np.array([[0.01, 0., 0., 4.], [0., 0.01, 0., 4.],
                           [0., 0., 0., 1.], [0., 0., 0., 1.]])
    metas = [
        dict(lidar2img=[projection] * num_cams,
             img_shape=[(8, 8, 3)] * num_cams)
        for _ in range(batch_size)
    ]
    return lidar, camera, metas


def check_predictions(outputs, num_layers, batch_size):
    cls, boxes = outputs[:2]
    expected = (num_layers, batch_size, NUM_TEST_QUERIES)
    assert cls.shape == expected + (10,), cls.shape
    assert boxes.shape == expected + (10,), boxes.shape
    assert torch.isfinite(cls).all() and torch.isfinite(boxes).all()


def check_detection_backward(model, lidar, camera, metas, attentions):
    model.train()
    boxes = [
        LiDARInstance3DBoxes([[1., 2., 0., 1.5, 4., 1.6, 0., 0., 0.]], box_dim=9)
        for _ in metas
    ]
    labels = [torch.tensor([0]) for _ in metas]
    losses = model.forward_pts_train(lidar, camera, None, boxes, labels, metas)
    loss = sum(losses.values())
    assert torch.isfinite(loss)
    loss.backward()
    norms = []
    for attention in attentions:
        parameters = list(attention.query_router.parameters())
        parameters.append(attention.modality_fusion_layer[0].weight)
        for parameter in parameters:
            grad = parameter.grad
            assert grad is not None and torch.isfinite(grad).all()
            assert torch.count_nonzero(grad) > 0
        norm = attention.query_router.mlp[-1].weight.grad.norm()
        norms.append(round(norm.item(), 4))
    print(f'  Detection backward: router output weight gradient norms {norms}')


def verify_model(cfg, state, mode, reference=None):
    torch.manual_seed(2026)
    model = build_from_checkpoint(cfg, state, mode)
    head = model.pts_bbox_head
    attentions = [layer.attentions[1] for layer in head.transformer.decoder.layers]
    assert all(attention.modality_fusion_layer[0].in_features == 512
               for attention in attentions)
    for embedding in (head.tgt_embed, head.refpoint_embed):
        embedding.weight = torch.nn.Parameter(
            embedding.weight[:NUM_TEST_QUERIES].detach().clone())

    torch.manual_seed(2026)
    lidar, camera, metas = synthetic_inputs(attentions[0])
    model.eval()
    with torch.no_grad():
        clean = head(lidar, camera, None, metas)
        check_predictions(clean, len(attentions), len(metas))
        if mode == 'none':
            assert all(attention.query_router is None for attention in attentions)
            if reference is not None:
                assert all(torch.equal(a, b) for a, b in zip(clean[:2], reference))
            return clean[:2]

        # Compare per-layer routing under identical synthetic clean/failure inputs.
        zero_lidar = [torch.zeros_like(level) for level in lidar]
        zero_camera = [torch.zeros_like(level) for level in camera]
        cases = {
            'clean': (lidar, camera),
            'LiDAR zeroed': (zero_lidar, camera),
            'camera zeroed': (lidar, zero_camera),
            'both zeroed': (zero_lidar, zero_camera),
            'LiDAR missing': (None, camera),
            'camera missing': (lidar, None),
        }
        for name, (pts_feats, img_feats) in cases.items():
            if name != 'clean':
                out = head(pts_feats, img_feats, None, metas)
                check_predictions(out, len(attentions), len(metas))
            percentages = []
            for attention in attentions:
                snapshot = attention.query_router.last_routing
                for key, channels in (('logits', 3), ('probabilities', 3), ('weights', 2)):
                    tensor = snapshot[key]
                    assert tensor.shape == (len(metas), NUM_TEST_QUERIES, channels)
                    assert not tensor.requires_grad and torch.isfinite(tensor).all()
                rates = snapshot['route_percentages']
                torch.testing.assert_close(rates.sum(), rates.new_tensor(100.))
                percentages.append([round(rate, 1) for rate in rates.tolist()])
            print(f'  {name} route percentages by layer (LiDAR, camera, both): {percentages}')
    check_detection_backward(model, lidar, camera, metas, attentions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path,
                        help='Original multimodal checkpoint (defaults to AQR load_from)')
    args = parser.parse_args()
    get_logger('mmcv').setLevel(logging.WARNING)
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    verify_router()

    cfg = Config.fromfile(AQR)
    checkpoint_path = args.checkpoint or ROOT / cfg.load_from
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint.get('state_dict', checkpoint)
    reference = verify_model(Config.fromfile(BASELINE), state, 'none')
    for mode in ('none', 'hard', 'soft'):
        cfg.merge_from_dict({
            f'{ATTENTION}.routing_mode': mode,
            f'{ATTENTION}.routing_diagnostics': mode != 'none',
        })
        print(f'{mode}:')
        verify_model(cfg, state, mode, reference)
        if mode == 'none':
            print('  Strict checkpoint load and exact baseline predictions passed.')
    print('All MAFS+AQR checks passed.')


if __name__ == '__main__':
    main()
