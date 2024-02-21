_base_ = './varibles.py'
input_modality = dict(use_lidar=True, use_camera=True)
backend_args = None
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
data_root = _base_.data_root
class_names = _base_.class_names

inter_channel = 256
voxel_size = [0.075, 0.075, 0.2]

model = dict(
    type='MFDetector',
    #use_grid_mask=True,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=False),
    
    img_backbone=dict(
        type='mmdet.SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=[1, 2, 3],
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa: E501
        )),
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[192, 384, 768],
        out_channels=inter_channel,
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        upsample_cfg=dict(mode='bilinear', align_corners=False)),
    
    # img_backbone=dict(
    #     type='VoVNetCP',
    #     spec_name='V-99-eSE',
    #     norm_eval=True,
    #     frozen_stages=-1,
    #     input_ch=3,
    #     out_features=('stage4','stage5',)),
    # img_neck=dict(
    #     type='CPFPN',
    #     in_channels=[768, 1024],
    #     out_channels=256,
    #     num_outs=2),
    pts_bbox_head=dict(
        type='MFFusionHead',
        use_camera=True,
        num_proposals=200,
        in_channels=512,
        hidden_channel=inter_channel,
        num_classes= len(class_names),
        nms_kernel_size=3,
        bn_momentum=0.1,
        #query denoising
        noise_scale=0.5,
        noise_trans=0.0,
        #center_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.1, batch_first=True),
        center_decoder_cfg = dict(
            type='mmdet.models.DeformableDetrTransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=inter_channel, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=inter_channel, num_levels=3, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=inter_channel,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
        ),
        decoder_cfg=dict(
            type='MFDeformableDetrTransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=inter_channel, num_heads=8, dropout=0.1),
            cls_cross_attn_cfg=dict(embed_dims=inter_channel, num_heads=8, dropout=0.1),
            box_cross_attn_cfg=dict(embed_dims=inter_channel, num_levels=3, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=inter_channel,
                feedforward_channels=1024,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
        ),
        center_head = dict( center=[2, 2] ),
        common_heads=dict(
             height=[1, 2], dim=[3, 2], rot=[2, 2], vel=[2, 2] ), #center=[2, 2],
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-54.0, -54.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=10),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        # loss_cls=dict(
        #     type='SoftFocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     reduction='mean',
        #     loss_weight=1.0),
        loss_frontmap=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=True,
            # gamma=2.0,
            # alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        loss_iou=dict(type='IoU3DLoss', loss_weight=1.0),
    ),
)

#db_sampler= _base_.db_sampler
train_pipeline = [
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='ImageAug3D',
        final_dim=[640, 1600],
        resize_lim=[0.94, 1.25],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=True,
        is_train=True),
    dict(type='ObjectSample', db_sampler = _base_.db_sampler ),
    dict(
        type='MFFusionGlobalRotScaleTrans',
        scale_ratio_range=[0.9, 1.1],
        rot_range=[-0.78539816, 0.78539816],
        translation_std=0.5),
    dict(type='MFFusionRandomFlip3D'),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        backend_args=backend_args),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=9,
        load_dim=5,
        use_dim=5,
        pad_empty_sweeps=True,
        remove_close=True,
        backend_args=backend_args),
    dict(
        type='BEVLoadMultiViewImageFromFiles',
        to_float32=True,
        color_type='color',
        backend_args=backend_args),
    dict(
        type='ImageAug3D',
        final_dim=[640, 1600],
        resize_lim=[0.94, 1.25],
        bot_pct_lim=[0.0, 0.0],
        rot_lim=[0.0, 0.0],
        rand_flip=False,
        is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'cam2img', 'ori_cam2img', 'lidar2cam', 'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size= _base_.train_batch_size,
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
            modality=input_modality)
            )
    )
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        ))
test_dataloader = val_dataloader

# learning rate
lr = 0.0001
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    # dict(
    #     type='CosineAnnealingLR',
    #     T_max=8,
    #     eta_min=lr * 8,
    #     begin=0,
    #     end=8,
    #     by_epoch=True,
    #     convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=6,
        eta_min=1e-8,
        begin=0,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    # dict(
    #     type='CosineAnnealingMomentum',
    #     T_max=0,
    #     eta_min=0.85 / 0.95,
    #     begin=0,
    #     end=8,
    #     by_epoch=True,
    #     convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=0,
        eta_min=1,
        begin=0,
        end=6,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=2)
val_cfg = dict()
test_cfg = dict()

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone':  dict(lr_mult=0.1),        #dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    clip_grad=dict(max_norm=35, norm_type=2))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (4 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size= 4*_base_.train_batch_size )

default_hooks = dict(
   logger=dict(type='LoggerHook', interval=50),
   checkpoint=dict(type='CheckpointHook', interval=1))
custom_hooks = [dict(type='DisableObjectSampleHook', disable_after_epoch=4)]

# vis_backends = [
#     dict(type='LocalVisBackend'),
#     #dict(type='TensorboardVisBackend'),
#     dict(type='WandbVisBackend',
#          init_kwargs={
#             'project': _base_.wandb_project,
#             'name': _base_.wandb_group,
#          },
#          #log_checkpoint=True,
#          )
# ]
# visualizer = dict(
#     type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

# load_from ='/mnt/data/users/chao.liu/checkpoints/basic_checkpoint/basic_lidar.pth'
