_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/neu_seg.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

custom_imports = dict(
    imports=['projects.industrial_defect_seg.datasets.neu_seg_dataset'],
    allow_failed_imports=False
)

crop_size = (200, 200)

data_preprocessor = dict(
    size=crop_size
)

checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/'
    'v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'
)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint
        )
    ),
    decode_head=dict(
        num_classes=4,
        norm_cfg=dict(type='BN', requires_grad=True)
    ),
    test_cfg=dict(mode='whole')
)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00006,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0)
        }
    )
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=200
    ),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        by_epoch=False,
        begin=200,
        end=20000
    )
]

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=500
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False
    ),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        max_keep_ckpts=3,
        save_best='mIoU',
        rule='greater'
    )
)

randomness = dict(
    seed=42,
    deterministic=False
)

work_dir = './work_dirs/comparison/neu/segformer_b0_neu_seg_20k'
