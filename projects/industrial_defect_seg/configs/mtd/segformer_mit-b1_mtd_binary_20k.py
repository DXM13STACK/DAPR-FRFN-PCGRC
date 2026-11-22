_base_ = './segformer_mit-b0_mtd_binary_20k.py'

model = dict(
    backbone=dict(
        _delete_=True,
        type='MixVisionTransformer',
        in_channels=3,
        embed_dims=64,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=(
                'https://download.openmmlab.com/mmsegmentation/'
                'v0.5/pretrain/segformer/'
                'mit_b1_20220624-02e5a6a1.pth'
            ),
        ),
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
    ),
)

randomness = dict(seed=42)

work_dir = 'work_dirs/comparison/mtd/segformer_mit-b1_mtd_binary_20k'
