_base_ = [
    './segformer_mit-b0_dapr_frfn_mtd_binary_20k.py'
]

custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset',
        'projects.industrial_defect_seg.models.dapr_frfn_pcgrc_segformer_head',
    ],
    allow_failed_imports=False,
)

model = dict(
    decode_head=dict(
        type='DAPRFRFNPCGRCSegformerHead',
        pcgrc_eps=1e-6,
        pcgrc_gate_reduction=16,
        pcgrc_gamma_init=0.1,
    )
)


train_dataloader = dict(
    batch_size=4,
)

work_dir = (
    'work_dirs/ablation/mtd/segformer_b0_dapr_frfn_pcgrc_mtd_binary_20k_bs4'
)

load_from = None
resume = False
