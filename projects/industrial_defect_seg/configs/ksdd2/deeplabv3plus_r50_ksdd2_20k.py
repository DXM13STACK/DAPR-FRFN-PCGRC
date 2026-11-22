_base_ = [
    '../../../../configs/deeplabv3plus/'
    'deeplabv3plus_r50-d8_4xb2-80k_cityscapes-512x1024.py'
]


custom_imports = dict(
    imports=[
        'projects.industrial_defect_seg.datasets.ksdd2_dataset'
    ],
    allow_failed_imports=False
)


dataset_type = 'KSDD2Dataset'
data_root = 'data/KSDD2/'


crop_size = (256, 640)


model = dict(

    data_preprocessor=dict(
        size=crop_size
    ),

    decode_head=dict(
        num_classes=2,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
    ),

    auxiliary_head=dict(
        num_classes=2,

        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.4
        )
    )
)


train_pipeline = [

    dict(
        type='LoadImageFromFile'
    ),

    dict(
        type='LoadAnnotations'
    ),

    dict(
        type='Resize',
        scale=(640, 256),
        keep_ratio=False
    ),

    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal'
    ),

    dict(
        type='PackSegInputs'
    )
]


test_pipeline = [

    dict(
        type='LoadImageFromFile'
    ),

    dict(
        type='Resize',
        scale=(640, 256),
        keep_ratio=False
    ),

    dict(
        type='LoadAnnotations'
    ),

    dict(
        type='PackSegInputs'
    )
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

    type='AmpOptimWrapper',

    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.9,
        weight_decay=0.0005
    ),

    loss_scale='dynamic'
)


param_scheduler = [

    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
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


work_dir = 'work_dirs/comparison/ksdd2/deeplabv3plus_r50_ksdd2_20k'
