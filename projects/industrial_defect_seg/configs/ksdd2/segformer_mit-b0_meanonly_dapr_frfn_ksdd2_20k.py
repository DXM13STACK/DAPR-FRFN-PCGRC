_base_ = [
    './segformer_mit-b0_ksdd2_20k.py'
]


custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset',
        'projects.industrial_defect_seg.models.meanonly_dapr_frfn_segformer_head',
    ],
    allow_failed_imports=False
)


model = dict(
    decode_head=dict(
        type='MeanOnlyDAPRFRFNSegformerHead',

        in_channels=[
            32,
            64,
            160,
            256
        ],

        in_index=[
            0,
            1,
            2,
            3
        ],

        channels=256,

        dropout_ratio=0.1,

        num_classes=2,

        norm_cfg=dict(
            type='SyncBN',
            requires_grad=True
        ),

        align_corners=False,

        frfn_mlp_ratio=4.0,

        dapr_reduction=16,

        dapr_partial_div=4,

        dapr_gamma_init=0.01,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        ),
    )
)


work_dir = (
    'work_dirs/ablation/ksdd2/segformer_b0_meanonly_dapr_frfn_ksdd2_20k'
)
