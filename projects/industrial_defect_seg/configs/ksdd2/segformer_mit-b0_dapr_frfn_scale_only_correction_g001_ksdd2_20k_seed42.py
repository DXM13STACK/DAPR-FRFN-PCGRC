_base_ = [
    './segformer_mit-b0_dapr_frfn_pcgrc_g001_ksdd2_20k.py'
]

custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset',
        'projects.industrial_defect_seg.models.'
        'dapr_frfn_scale_only_correction_segformer_head',
    ],
    allow_failed_imports=False,
)

model = dict(
    decode_head=dict(
        type='DAPRFRFNScaleOnlyCorrectionSegformerHead',
        pcgrc_gamma_init=0.01,
    )
)

randomness = dict(seed=42)

work_dir = (
    'work_dirs/ablation/ksdd2/'
    'segformer_b0_dapr_frfn_scale_only_correction_g001_ksdd2_20k_seed42'
)

load_from = None
resume = False
