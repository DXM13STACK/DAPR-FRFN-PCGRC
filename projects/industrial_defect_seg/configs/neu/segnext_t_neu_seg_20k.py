_base_ = '../ksdd2/segnext_t_ksdd2_20k.py'

custom_imports = dict(
    imports=['projects.industrial_defect_seg.datasets.neu_seg_dataset'],
    allow_failed_imports=False,
)

dataset_type = 'NEUSegDataset'
data_root = 'data/NEU_Seg'

crop_size = (200, 200)


model = dict(
    data_preprocessor=dict(
        size=crop_size,
    ),
    decode_head=dict(
        num_classes=4,
    ),
)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal',
    ),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction='vertical',
    ),
    dict(
        type='RandomRotate',
        prob=0.5,
        degree=10,
        pad_val=0,
        seg_pad_val=255,
    ),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    _delete_=True,
    batch_size=12,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='InfiniteSampler',
        shuffle=True,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/train.txt',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/val.txt',
        data_prefix=dict(
            img_path='images/training',
            seg_map_path='annotations/training',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    _delete_=True,
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='splits/test.txt',
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='annotations/test',
        ),
        pipeline=test_pipeline,
    ),
)

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
)

test_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU', 'mDice'],
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=500,
)

randomness = dict(seed=42)

work_dir = 'work_dirs/comparison/neu/segnext_t_neu_seg_20k'
