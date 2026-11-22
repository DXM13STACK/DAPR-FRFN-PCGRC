_base_ = [
    '../../../../configs/bisenetv2/'
    'bisenetv2_fcn_4xb4-160k_cityscapes-1024x1024.py'
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


norm_cfg = dict(
    type='SyncBN',
    requires_grad=True
)

model = dict(

    data_preprocessor=dict(
        size=crop_size
    ),

    decode_head=dict(
        num_classes=2
    ),

    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=16,
            channels=16,
            num_convs=2,
            num_classes=2,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        ),
        dict(
            type='FCNHead',
            in_channels=32,
            channels=64,
            num_convs=2,
            num_classes=2,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        ),
        dict(
            type='FCNHead',
            in_channels=64,
            channels=256,
            num_convs=2,
            num_classes=2,
            in_index=3,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        ),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=1024,
            num_convs=2,
            num_classes=2,
            in_index=4,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0
            )
        )
    ]
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
        type='SGD',
        lr=0.05,
        momentum=0.9,
        weight_decay=0.0005
    )
)


param_scheduler = [
    dict(
        type='LinearLR',
        by_epoch=False,
        start_factor=0.1,
        begin=0,
        end=1000
    ),
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=1000,
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

work_dir = 'work_dirs/comparison/ksdd2/bisenetv2_ksdd2_20k'
