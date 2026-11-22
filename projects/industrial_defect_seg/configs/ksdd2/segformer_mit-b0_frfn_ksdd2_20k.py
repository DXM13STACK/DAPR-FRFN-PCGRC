_base_ = [
    './segformer_mit-b0_ksdd2_20k.py'
]


custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset',
        'projects.industrial_defect_seg.models.frfn_segformer_head',
    ],
    allow_failed_imports=False,
)


model = dict(
    decode_head=dict(
        type='FRFNSegformerHead',


        frfn_mlp_ratio=4.0,
    )
)


work_dir = (
    'work_dirs/ablation/ksdd2/segformer_b0_frfn_ksdd2_20k'
)
