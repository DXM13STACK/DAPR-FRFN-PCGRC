_base_ = [
    './segformer_mit-b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4.py'
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

train_dataloader = dict(batch_size=4)

randomness = dict(seed=42)

work_dir = (
    'work_dirs/ablation/mtd/'
    'segformer_b0_dapr_frfn_scale_only_correction_g001_mtd_binary_20k_bs4_seed42'
)

load_from = None
resume = False
