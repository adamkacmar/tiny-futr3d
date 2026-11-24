import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryRouter(nn.Module):
    """Weight the sampled LiDAR and camera features of each object query."""

    def __init__(self, embed_dims, mode, hidden_dims=64, diagnostics=False):
        super().__init__()
        if mode not in ('hard', 'soft'):
            raise ValueError(f'Unsupported query routing mode: {mode}')
        if hidden_dims <= 0:
            raise ValueError('Router hidden dimension must be positive')
        self.mode = mode
        self.diagnostics = diagnostics
        self.last_routing = None
        self.mlp = nn.Sequential(
            nn.Linear(2 * embed_dims, hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, 3),
        )

    def forward(self, lidar, camera):
        logits = self.mlp(torch.cat((lidar, camera), dim=-1))
        use_soft_weights = self.mode == 'soft' or self.training
        if use_soft_weights or self.diagnostics:
            # Keep the softmax and straight-through surrogate stable under AMP.
            probability_dtype = (torch.float32 if logits.dtype in
                                 (torch.float16, torch.bfloat16)
                                 else logits.dtype)
            probabilities = logits.softmax(dim=-1, dtype=probability_dtype)
        if use_soft_weights:
            lidar_only, camera_only, both = probabilities.unbind(dim=-1)
            weights = torch.stack((lidar_only + both, camera_only + both),
                                  dim=-1)
        if self.mode == 'hard' or self.diagnostics:
            selected = logits.argmax(dim=-1)
        if self.mode == 'hard':
            # Class order is LiDAR only, camera only, both.
            hard_weights = torch.stack(
                (selected != 1, selected != 0), dim=-1).to(logits.dtype)
            if self.training:
                # The parenthesized difference is exactly zero in the forward
                # pass and passes the soft weights' gradient in backward.
                weights = hard_weights.to(weights.dtype) + (
                    weights - weights.detach())
            else:
                weights = hard_weights

        if self.diagnostics:
            self.last_routing = {
                'logits': logits.detach(),
                'probabilities': probabilities.detach(),
                'weights': weights.detach(),
                'route_percentages': (
                    F.one_hot(selected, num_classes=3).float().mean((0, 1))
                    * 100),
            }
        lidar_weight, camera_weight = weights.split(1, dim=-1)
        return (lidar * lidar_weight.to(lidar.dtype),
                camera * camera_weight.to(camera.dtype))
