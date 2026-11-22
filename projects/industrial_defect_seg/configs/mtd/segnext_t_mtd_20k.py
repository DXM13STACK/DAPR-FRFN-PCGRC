_base_ = '../ksdd2/segnext_t_ksdd2_20k.py'

dataset_type = 'BaseSegDataset'
data_root = 'data/MTD_binary/'

metainfo = dict(
    classes=('background', 'defect'),
    palette=[
        [0, 0, 0],
        [255, 255, 255],
    ],
)

crop_size = (512, 512)

model = dict(
    data_preprocessor=dict(
        size=crop_size,
    ),
    decode_head=dict(
        num_classes=2,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=True,
    ),
    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal',
    ),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='Resize',
        scale=(512, 512),
        keep_ratio=True,
    ),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='InfiniteSampler',
        shuffle=True,
    ),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train',
        ),
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val',
        ),
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False,
    ),
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        img_suffix='.jpg',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='annotations/test',
        ),
        pipeline=test_pipeline,
    ),
)

randomness = dict(seed=42)

work_dir = 'work_dirs/comparison/mtd/segnext_t_mtd_20k'
