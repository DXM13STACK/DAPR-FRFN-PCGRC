_base_ = [
    './segformer_mit-b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4.py'
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

work_dir = (
    'work_dirs/ablation/mtd/segformer_b0_dapr_frfn_naive_correction_mtd_binary_20k_bs4'
)
