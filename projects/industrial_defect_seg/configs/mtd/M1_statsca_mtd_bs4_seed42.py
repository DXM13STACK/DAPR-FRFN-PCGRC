checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'

crop_size = (512, 512)

custom_imports = {'imports': ['projects.industrial_defect_seg.models.stats_ablation_segformer_head'],
 'allow_failed_imports': False}

data_preprocessor = {'bgr_to_rgb': True,
 'mean': [123.675, 116.28, 103.53],
 'pad_val': 0,
 'seg_pad_val': 255,
 'size': (512, 512),
 'std': [58.395, 57.12, 57.375],
 'type': 'SegDataPreProcessor'}

data_root = 'data/MTD_binary/'

dataset_type = 'BaseSegDataset'

default_hooks = {'checkpoint': {'by_epoch': False,
                'interval': 500,
                'max_keep_ckpts': 3,
                'rule': 'greater',
                'save_best': 'mIoU',
                'type': 'CheckpointHook'},
 'logger': {'interval': 50, 'log_metric_by_epoch': False, 'type': 'LoggerHook'},
 'param_scheduler': {'type': 'ParamSchedulerHook'},
 'sampler_seed': {'type': 'DistSamplerSeedHook'},
 'timer': {'type': 'IterTimerHook'},
 'visualization': {'type': 'SegVisualizationHook'}}

default_scope = 'mmseg'

env_cfg = {'cudnn_benchmark': True,
 'dist_cfg': {'backend': 'nccl'},
 'mp_cfg': {'mp_start_method': 'fork', 'opencv_num_threads': 0}}

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

launcher = 'none'

load_from = None

log_level = 'INFO'

log_processor = {'by_epoch': False}

metainfo = {'classes': ('background', 'defect'), 'palette': [[0, 0, 0], [255, 255, 255]]}

model = {'auxiliary_head': None,
 'backbone': {'attn_drop_rate': 0.0,
              'drop_path_rate': 0.1,
              'drop_rate': 0.0,
              'embed_dims': 32,
              'in_channels': 3,
              'init_cfg': {'checkpoint': 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth',
                           'type': 'Pretrained'},
              'mlp_ratio': 4,
              'num_heads': [1, 2, 5, 8],
              'num_layers': [2, 2, 2, 2],
              'num_stages': 4,
              'out_indices': (0, 1, 2, 3),
              'patch_sizes': [7, 3, 3, 3],
              'qkv_bias': True,
              'sr_ratios': [8, 4, 2, 1],
              'type': 'MixVisionTransformer'},
 'data_preprocessor': {'bgr_to_rgb': True,
                       'mean': [123.675, 116.28, 103.53],
                       'pad_val': 0,
                       'seg_pad_val': 255,
                       'size': (512, 512),
                       'std': [58.395, 57.12, 57.375],
                       'type': 'SegDataPreProcessor'},
 'decode_head': {'align_corners': False,
                 'channels': 256,
                 'dropout_ratio': 0.1,
                 'frfn_mlp_ratio': 4.0,
                 'in_channels': [32, 64, 160, 256],
                 'in_index': [0, 1, 2, 3],
                 'loss_decode': {'loss_weight': 1.0, 'type': 'CrossEntropyLoss', 'use_sigmoid': False},
                 'norm_cfg': {'requires_grad': True, 'type': 'SyncBN'},
                 'num_classes': 2,
                 'stats_reduction': 16,
                 'type': 'StatsCAFRFNSegformerHead'},
 'pretrained': None,
 'test_cfg': {'mode': 'whole'},
 'train_cfg': {},
 'type': 'EncoderDecoder'}

norm_cfg = {'requires_grad': True, 'type': 'SyncBN'}

optim_wrapper = {'loss_scale': 'dynamic',
 'optimizer': {'betas': (0.9, 0.999), 'lr': 6e-05, 'type': 'AdamW', 'weight_decay': 0.01},
 'paramwise_cfg': {'custom_keys': {'head': {'lr_mult': 10.0},
                                   'norm': {'decay_mult': 0.0},
                                   'pos_block': {'decay_mult': 0.0}}},
 'type': 'AmpOptimWrapper'}

optimizer = {'lr': 0.01, 'momentum': 0.9, 'type': 'SGD', 'weight_decay': 0.0005}

param_scheduler = [{'begin': 0, 'by_epoch': False, 'end': 1500, 'start_factor': 1e-06, 'type': 'LinearLR'},
 {'begin': 1500, 'by_epoch': False, 'end': 20000, 'eta_min': 0.0, 'power': 1.0, 'type': 'PolyLR'}]

randomness = {'seed': 42}

resume = False

test_cfg = {'type': 'TestLoop'}

test_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'images/test', 'seg_map_path': 'annotations/test'},
             'data_root': 'data/MTD_binary/',
             'img_suffix': '.jpg',
             'metainfo': {'classes': ('background', 'defect'), 'palette': [[0, 0, 0], [255, 255, 255]]},
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (512, 512), 'type': 'Resize'},
                          {'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'reduce_zero_label': False,
             'seg_map_suffix': '.png',
             'type': 'BaseSegDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

test_evaluator = {'iou_metrics': ['mIoU', 'mDice'], 'type': 'IoUMetric'}

test_pipeline = [{'type': 'LoadImageFromFile'},
 {'keep_ratio': True, 'scale': (512, 512), 'type': 'Resize'},
 {'type': 'LoadAnnotations'},
 {'type': 'PackSegInputs'}]

train_cfg = {'max_iters': 20000, 'type': 'IterBasedTrainLoop', 'val_interval': 500}

train_dataloader = {'batch_size': 4,
 'dataset': {'data_prefix': {'img_path': 'images/train', 'seg_map_path': 'annotations/train'},
             'data_root': 'data/MTD_binary/',
             'img_suffix': '.jpg',
             'metainfo': {'classes': ('background', 'defect'), 'palette': [[0, 0, 0], [255, 255, 255]]},
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'type': 'LoadAnnotations'},
                          {'keep_ratio': True, 'scale': (512, 512), 'type': 'Resize'},
                          {'direction': 'horizontal', 'prob': 0.5, 'type': 'RandomFlip'},
                          {'type': 'PackSegInputs'}],
             'reduce_zero_label': False,
             'seg_map_suffix': '.png',
             'type': 'BaseSegDataset'},
 'num_workers': 0,
 'persistent_workers': False,
 'sampler': {'shuffle': True, 'type': 'InfiniteSampler'}}

train_pipeline = [{'type': 'LoadImageFromFile'},
 {'type': 'LoadAnnotations'},
 {'keep_ratio': True, 'scale': (512, 512), 'type': 'Resize'},
 {'direction': 'horizontal', 'prob': 0.5, 'type': 'RandomFlip'},
 {'type': 'PackSegInputs'}]

tta_model = {'type': 'SegTTAModel'}

tta_pipeline = [{'backend_args': None, 'type': 'LoadImageFromFile'},
 {'transforms': [[{'keep_ratio': True, 'scale_factor': 0.5, 'type': 'Resize'},
                  {'keep_ratio': True, 'scale_factor': 0.75, 'type': 'Resize'},
                  {'keep_ratio': True, 'scale_factor': 1.0, 'type': 'Resize'},
                  {'keep_ratio': True, 'scale_factor': 1.25, 'type': 'Resize'},
                  {'keep_ratio': True, 'scale_factor': 1.5, 'type': 'Resize'},
                  {'keep_ratio': True, 'scale_factor': 1.75, 'type': 'Resize'}],
                 [{'direction': 'horizontal', 'prob': 0.0, 'type': 'RandomFlip'},
                  {'direction': 'horizontal', 'prob': 1.0, 'type': 'RandomFlip'}],
                 [{'type': 'LoadAnnotations'}],
                 [{'type': 'PackSegInputs'}]],
  'type': 'TestTimeAug'}]

val_cfg = {'type': 'ValLoop'}

val_dataloader = {'batch_size': 1,
 'dataset': {'data_prefix': {'img_path': 'images/val', 'seg_map_path': 'annotations/val'},
             'data_root': 'data/MTD_binary/',
             'img_suffix': '.jpg',
             'metainfo': {'classes': ('background', 'defect'), 'palette': [[0, 0, 0], [255, 255, 255]]},
             'pipeline': [{'type': 'LoadImageFromFile'},
                          {'keep_ratio': True, 'scale': (512, 512), 'type': 'Resize'},
                          {'type': 'LoadAnnotations'},
                          {'type': 'PackSegInputs'}],
             'reduce_zero_label': False,
             'seg_map_suffix': '.png',
             'type': 'BaseSegDataset'},
 'num_workers': 4,
 'persistent_workers': True,
 'sampler': {'shuffle': False, 'type': 'DefaultSampler'}}

val_evaluator = {'iou_metrics': ['mIoU', 'mDice'], 'type': 'IoUMetric'}

vis_backends = [{'type': 'LocalVisBackend'}]

visualizer = {'name': 'visualizer', 'type': 'SegLocalVisualizer', 'vis_backends': [{'type': 'LocalVisBackend'}]}

work_dir = 'work_dirs/mtd_dapr_internal_20261016/M1_statsca_mtd_bs4_seed42'
