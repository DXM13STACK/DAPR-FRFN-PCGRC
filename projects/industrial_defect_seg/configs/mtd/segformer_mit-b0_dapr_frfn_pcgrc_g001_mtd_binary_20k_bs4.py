_base_ = [
    './segformer_mit-b0_dapr_frfn_pcgrc_mtd_binary_20k_bs4.py'
]

model = dict(
    decode_head=dict(
        pcgrc_gamma_init=0.01,
    )
)

work_dir = (
    'work_dirs/comparison/mtd/segformer_b0_dapr_frfn_pcgrc_g001_mtd_binary_20k_bs4'
)

load_from = None
resume = False
