

_base_ = [
    '../../../../configs/swin/'
    'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512.py'
]


custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset'
    ],
    allow_failed_imports=False
)

dataset_type = 'KSDD2Dataset'
data_root = 'data/KSDD2/'


crop_size = (640, 256)


model = dict(

    data_preprocessor=dict(
        size=crop_size
    ),

    backbone=dict(
        with_cp=False
    ),

    decode_head=dict(
        num_classes=2
    ),

    auxiliary_head=dict(
        num_classes=2
    )
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),

    dict(
        type='Resize',
        scale=(256, 640),
        keep_ratio=False
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal'
    ),

    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),

    dict(
        type='Resize',
        scale=(256, 640),
        keep_ratio=False
    ),

    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type='InfiniteSampler',
        shuffle=True
    ),

    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,

        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train'
        ),

        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),

    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,

        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val'
        ),

        pipeline=test_pipeline
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,

    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),

    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,

        data_prefix=dict(
            img_path='images/test',
            seg_map_path='annotations/test'
        ),

        pipeline=test_pipeline
    )
)


val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mIoU',
        'mDice'
    ]
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=[
        'mIoU',
        'mDice'
    ]
)


train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=500
)

val_cfg = dict(
    type='ValLoop'
)

test_cfg = dict(
    type='TestLoop'
)


optim_wrapper = dict(
    _delete_=True,

    type='OptimWrapper',

    optimizer=dict(
        type='AdamW',
        lr=6e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),

    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(
                decay_mult=0.0
            ),
            'relative_position_bias_table': dict(
                decay_mult=0.0
            ),
            'norm': dict(
                decay_mult=0.0
            )
        }
    )
)


param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500
    ),

    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=20000,
        by_epoch=False
    )
]


default_hooks = dict(

    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=500,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3
    ),

    logger=dict(
        type='LoggerHook',
        interval=50
    )
)

log_processor = dict(
    by_epoch=False
)

randomness = dict(
    seed=42
)

work_dir = 'work_dirs/comparison/ksdd2/upernet_swin_t_ksdd2_20k'
