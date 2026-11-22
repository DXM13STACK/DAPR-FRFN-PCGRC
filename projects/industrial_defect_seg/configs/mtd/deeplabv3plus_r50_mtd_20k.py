_base_ = '../ksdd2/deeplabv3plus_r50_ksdd2_20k.py'

dataset_type = 'BaseSegDataset'
data_root = 'data/MTD_binary/'

metainfo = dict(
    classes=('background', 'defect'),
    palette=[[0, 0, 0], [255, 255, 255]],
)


model = dict(
    data_preprocessor=dict(
        size=(512, 512),
    ),
    decode_head=dict(
        num_classes=2,
    ),
    auxiliary_head=dict(
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
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/train',
            seg_map_path='annotations/train',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=metainfo,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/val',
            seg_map_path='annotations/val',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/test',
            seg_map_path='annotations/test',
        ),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        metainfo=metainfo,
        pipeline=test_pipeline,
    ),
)

randomness = dict(seed=42)

work_dir = 'work_dirs/comparison/mtd/deeplabv3plus_r50_mtd_20k'
