_base_ = [
    './segformer_mit-b0_dapr_frfn_pcgrc_ksdd2_20k.py'
]

model = dict(
    decode_head=dict(
        pcgrc_gamma_init=0.01,
    )
)

work_dir = (
    'work_dirs/comparison/ksdd2/segformer_b0_dapr_frfn_pcgrc_g001_ksdd2_20k'
)

load_from = None
resume = False
