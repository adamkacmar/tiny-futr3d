_base_ = ['./lidar_0075v_cam_vov_eval_clean.py']

# MMCV replaces lists during config inheritance. Repeat the two attention
# entries so only the cross-attention routing options differ from the baseline.
model = dict(
    pts_bbox_head=dict(
        transformer=dict(
            decoder=dict(
                transformerlayers=dict(
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='FUTR3DAttention',
                            use_lidar=True,
                            use_camera=True,
                            use_radar=False,
                            pc_range={{_base_.point_cloud_range}},
                            embed_dims=256,
                            routing_mode='hard',
                            routing_hidden_dims=64,
                            routing_diagnostics=False),
                    ])))))
