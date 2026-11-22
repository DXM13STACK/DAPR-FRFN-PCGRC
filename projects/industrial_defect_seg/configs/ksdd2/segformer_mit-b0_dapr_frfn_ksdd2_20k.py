_base_ = [
    './segformer_mit-b0_ksdd2_20k.py'
]

custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset',
        'projects.industrial_defect_seg.models.dapr_frfn_segformer_head',
    ],
    allow_failed_imports=False
)

model = dict(
    decode_head=dict(
        type='DAPRFRFNSegformerHead',
        frfn_mlp_ratio=4.0,
        dapr_reduction=16,
        dapr_partial_div=4,
        dapr_gamma_init=0.01,
    )
)

work_dir = (
    'work_dirs/ablation/ksdd2/segformer_b0_dapr_frfn_ksdd2_20k'
)
