_base_ = [
    './segformer_mit-b0_dapr_frfn_pcgrc_g001_ksdd2_20k.py'
]

custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset',
        'projects.industrial_defect_seg.models.dapr_frfn_pcgrc_segformer_head',
        'projects.industrial_defect_seg.models.dapr_frfn_naive_correction_segformer_head',
    ],
    allow_failed_imports=False,
)

model = dict(
    decode_head=dict(
        type='DAPRFRFNNaiveCorrectionSegformerHead',
        pcgrc_gamma_init=1.0,
    )
)

train_dataloader = dict(batch_size=8)
randomness = dict(seed=42)

work_dir = (
    'work_dirs/ablation/ksdd2/'
    'segformer_b0_dapr_frfn_naive_correction_ksdd2_20k_bs8_seed42'
)

load_from = None
resume = False
