_base_ = '../../../../configs/segformer/segformer_mit-b0_neu_seg_20k.py'

custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.neu_seg_dataset',
        'projects.industrial_defect_seg.models.dapr_frfn_segformer_head',
    ],
    allow_failed_imports=False,
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

load_from = None
resume = False

work_dir = './work_dirs/ablation/neu/segformer_b0_dapr_frfn_neu_seg_20k'
