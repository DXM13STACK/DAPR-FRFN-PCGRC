

_base_ = [
    './segformer_mit-b0_mtd_binary_20k.py'
]


norm_cfg = dict(
    type='SyncBN',
    requires_grad=True
)

model = dict(
    _delete_=True,

    type='EncoderDecoder',

    data_preprocessor=dict(
        type='SegDataPreProcessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
        size=(512, 512)
    ),

    pretrained=None,

    backbone=dict(
        type='BiSeNetV2',
        detail_channels=(64, 64, 128),
        semantic_channels=(16, 32, 64, 128),
        semantic_expansion_ratio=6,
        bga_channels=128,
        out_indices=(0, 1, 2, 3, 4),
        init_cfg=None,
        align_corners=False
    ),

    decode_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=0,
        channels=1024,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=1.0
        )
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
    ],

    train_cfg=dict(),

    test_cfg=dict(
        mode='whole'
    )
)


train_dataloader = dict(
    batch_size=8
)

val_dataloader = dict(
    batch_size=1
)

test_dataloader = dict(
    batch_size=1
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


train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=20000,
    val_interval=500
)

randomness = dict(
    seed=42
)

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


work_dir = 'work_dirs/comparison/mtd/bisenetv2_mtd_20k'
