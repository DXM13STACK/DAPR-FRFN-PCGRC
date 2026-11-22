_base_ = './segformer_mit-b0_dapr_frfn_neu_seg_20k.py'

custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.neu_seg_dataset',
        'projects.industrial_defect_seg.models.dapr_frfn_segformer_head',
        'projects.industrial_defect_seg.models.dapr_frfn_pcgrc_segformer_head',
    ],
    allow_failed_imports=False,
)

model = dict(
    decode_head=dict(
        type='DAPRFRFNPCGRCSegformerHead',
        pcgrc_gamma_init=0.01,
    )
)

randomness = dict(
    seed=42,
    deterministic=False,
)

work_dir = (
    'work_dirs/comparison/neu/segformer_b0_dapr_frfn_pcgrc_g001_neu_seg_20k'
)
