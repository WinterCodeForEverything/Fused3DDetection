import copy
import math
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.ops import furthest_point_sample, points_in_boxes_all
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
#from mmdet.models.utils import inverse_sigmoid
from mmengine.structures import InstanceData

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.registry import MODELS
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from .prediction_head import MFSeparateHead

from mmdet3d.structures import xywhr2xyxyr
from mmdet3d.structures.bbox_3d import BaseInstance3DBoxes, LiDARInstance3DBoxes
from mmdet3d.models.layers import nms_bev #VoteModule, build_sa_module
from mmdet3d.structures.ops.iou3d_calculator import bbox_overlaps_3d, bbox_overlaps_nearest_3d
# from mmdet3d.models.losses.chamfer_distance import chamfer_distance
from .utils import draw_bool_heatmap        #, weighted_furthest_point_sample

#from mmcv.cnn.bricks.transformer import  MultiheadAttention


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y

@MODELS.register_module()
class MFFusionHead(nn.Module):

    def __init__(self,
        num_proposals=200,
        in_channels=128 * 4,
        hidden_channel=128,
        num_classes=4,
        multistage_heatmap = 0,
        #query denoising
        noise_scale=1.0,
        noise_trans=0.0,
        center_attn_cfg = dict(),
        # config for Transformer
        center_decoder_cfg=dict(),
        decoder_cfg=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        center_head=dict(),
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.FocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_iou=dict(type='mmdet.GIoULoss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_frontmap=dict(type='mmdet.FocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
                 ) -> None:
        super(MFFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals_ori = num_proposals
        self.num_proposals = num_proposals
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.pc_range = train_cfg['point_cloud_range']
        self.multistage_heatmap = multistage_heatmap


        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)
        self.loss_heatmap = MODELS.build(loss_heatmap)
        self.loss_frontmap = MODELS.build(loss_frontmap)
        self.sampling = False

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pos_embed_channel = hidden_channel
        # self.front_tokens_threshold = front_tokens_threshold
    
        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        
        self.multiscale = True
        if self.multiscale:
            self.dconv = ConvModule(
                hidden_channel, hidden_channel, 
                stride=2, kernel_size=3, padding=1, bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            )
            self.dconv2 = ConvModule(
                hidden_channel, hidden_channel, 
                stride=2, kernel_size=3, padding=1, bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            )
        
        # frontmap_layers = []
        # frontmap_layers.append(
        #     ConvModule(
        #         hidden_channel,
        #         hidden_channel,
        #         kernel_size=1,
        #         padding=0,
        #         bias=bias,
        #         conv_cfg=dict(type='Conv2d'),
        #         norm_cfg=dict(type='BN2d'),
        #     ))
        # frontmap_layers.append(
        #     build_conv_layer(
        #         dict(type='Conv2d'),
        #         hidden_channel,
        #         1,
        #         kernel_size=1,
        #         padding=0,
        #         bias=bias,
        #     ))
        # self.frontmap_head = nn.Sequential(*frontmap_layers)

        heatmap_layers = []
        heatmap_layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        heatmap_layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*heatmap_layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        
        if self.multistage_heatmap != 0:
            self.multistage_heatmap_head = nn.ModuleList()
            for _ in range(self.multistage_heatmap):
                self.multistage_heatmap_head.append(copy.deepcopy(self.heatmap_head)) # just copy parameter values.
                

        #self.points_encoder = MODELS.build(points_encoder)
        #self.vote_module = VoteModule(**vote_module_cfg)
        #self.vote_aggregation = build_sa_module(vote_aggregation_cfg)

        # transformer decoder layers for object query with LiDAR feature
        self.bev_embedding = nn.Sequential(
            nn.Linear(hidden_channel * 2, hidden_channel),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel)
        )           

        # self.depth_num = 64
        # self.rv_embedding = nn.Sequential(
        #     nn.Linear(self.depth_num * 3, hidden_channel * 4),
        #     nn.ReLU(inplace=True),
        #     nn.Linear( hidden_channel * 4, hidden_channel)
        # )
        #self.center_decoder = MODELS.build(decoder_layer)
        #self.center_attn = MultiheadAttention(**center_attn_cfg)
        self.center_transformer_decoder = MODELS.build(center_decoder_cfg)
        self.center_head = MFSeparateHead( hidden_channel,
                                             center_head,
                                             conv_cfg=conv_cfg,
                                             norm_cfg=norm_cfg,
                                             bias=bias,
            )
        
        self.decoder = MODELS.build(decoder_cfg)

        # Prediction Head
        common_heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
        self.prediction_head = MFSeparateHead( hidden_channel,
                                             common_heads,
                                             conv_cfg=conv_cfg,
                                             norm_cfg=norm_cfg,
                                             bias=bias,
            )
        
        self.init_weights()
        self._init_assigner_sampler()

        self.bev_pos = self.coords_bev()
        if self.multiscale:
            self.bev_pos2 = self.coords_bev(downsample_factor=2)
            self.bev_pos4 = self.coords_bev(downsample_factor=4)
        
    @property
    def sample_mode(self):
        if self.training:
            sample_mode = self.train_cfg.sample_mode
        else:
            sample_mode = self.test_cfg.sample_mode
        assert sample_mode in ['vote', 'seed', 'random', 'spec']
        return sample_mode
        
    # def compute_seed_feature(self, seed_points, bev_feature, bev_pos ):
    #     # seed_points: (B, N, 3)
    #     # bev_feature: (B, C, M )
    #     # bev_pos: (B, M, 2)
    #     #=> seed_feature: (B, C, N)
        
    #     bev_feature = bev_feature.permute(0, 2, 1).contiguous()
    #     cfg = self.train_cfg if self.train_cfg else self.test_cfg
    #     voxel_size = seed_points.new_tensor(cfg['voxel_size']) 
               
    #     seed_xy = seed_points[..., :2].flip(-1)
    #     bev_seed_xy = (seed_xy - seed_xy.new_tensor(self.pc_range[:2])) / voxel_size[:2] / cfg['out_size_factor']
    #     bev_seed_yx = bev_seed_xy.flip(-1)
    #     bev_seed_yx_expand = bev_seed_yx.unsqueeze(2).repeat(1, 1, bev_pos.shape[1], 1)
    #     bev_pos_expand = bev_pos.unsqueeze(1).repeat(1, bev_seed_yx.shape[1], 1, 1)
        
        
    #     distance = F.mse_loss(bev_seed_yx_expand, bev_pos_expand, reduction='none').sum(-1)
    #     #print( distance.shape )
        
    #     weight = distance.softmax(dim=-1)
    #     seed_feature = (bev_feature.unsqueeze(1) * weight.unsqueeze(-1)).sum(dim=-2)
        
    #     return seed_feature.permute(0, 2, 1).contiguous()

    def coords_bev(self, downsample_factor=1):
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = torch.div(cfg['grid_size'][0], cfg['out_size_factor']*downsample_factor, rounding_mode='floor')
        y_size = torch.div(cfg['grid_size'][1], cfg['out_size_factor']*downsample_factor, rounding_mode='floor')
        meshgrid = [[0, y_size - 1, y_size], [0, x_size - 1, x_size]]
        # NOTE: modified
        batch_y, batch_x = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_y = batch_y + 0.5
        batch_x = batch_x + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1) # (1, H*W, 2)            
        return coord_base
    
    def pos2embed(self, pos, num_pos_feats=128, temperature=10000):
        scale = 2 * math.pi
        pos = pos * scale
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
        dim_t = 2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats + 1
        pos_x = pos[..., 0, None] / dim_t
        pos_y = pos[..., 1, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        posemb = torch.cat(( pos_x, pos_y ), dim=-1 )
        return posemb

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
    
    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]
   
    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            known_num = [t.size(0) for t in targets]
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])

            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            groups = min(self.scalar, self.num_query // max(known_num))
            known_indice = known_indice.repeat(groups, 1).view(-1)
            known_labels = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_labels_raw = labels.repeat(groups, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(groups, 1).view(-1)
            known_bboxs = boxes.repeat(groups, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()
            
            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:1] = (known_bbox_center[..., 0:1] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0])
                known_bbox_center[..., 1:2] = (known_bbox_center[..., 1:2] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1])
                known_bbox_center[..., 2:3] = (known_bbox_center[..., 2:3] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2])
                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = sum(self.num_classes)

            single_pad = int(max(known_num))
            pad_size = int(single_pad * groups)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(groups)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(groups):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == groups - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'known_labels_raw': known_labels_raw,
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _bev_query_embed(self, ref_points):
        bev_embeds = self.bev_embedding(self.pos2embed(ref_points, num_pos_feats=self.pos_embed_channel))
        return bev_embeds

    def query_embed(self, ref_points, img_metas):
        bev_embeds = self._bev_query_embed(ref_points)
        return bev_embeds
    
    @torch.no_grad()  
    def get_front_points_mask( self, points, front_bev_mask ):
        
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        bev_size = torch.div(points.new_tensor(cfg['grid_size'][:2]), cfg['out_size_factor'], rounding_mode='floor').long()
        voxel_size = points.new_tensor(cfg['voxel_size'])
        bev_idx = torch.div((points[..., :2]-points.new_tensor(self.pc_range[:2])), 
            voxel_size[:2] * cfg['out_size_factor'],
              rounding_mode='floor').long()
        assert  ( bev_idx >= 0).all() & (bev_idx <= bev_size).all() 
        bev_idx[..., 0] = torch.clamp( bev_idx[..., 0], min = 0, max = bev_size[0] - 1 )
        bev_idx[..., 1] = torch.clamp( bev_idx[..., 1], min = 0, max = bev_size[1] - 1 )
        idx = bev_idx[..., 1] * bev_size[0] + bev_idx[..., 0]

        #front_bev_mask = front_bev_mask.view( front_bev_mask.shape[0], -1 )
        front_points_mask = torch.gather( front_bev_mask, dim = -1, index = idx )
        #front_points_mask = torch.ones_like( idx, dtype=torch.bool )
        #front_idx = idx.gather( 1, front_idx )

        #front_points_mask = (idx.unsqueeze(-1) == front_idx.unsqueeze(-2)).any(dim = -1)    

        #front_points_mask = None #frontboolmap.gather( index = idx, dim = -1 )
        return front_points_mask
    
    @torch.no_grad()    
    def get_img_tokens_reference_points( self, front_points, img_feats, img_metas ):
        _, N, _ = front_points.shape
        B, V, C, H, W = img_feats.shape
        image_shape = img_metas[0]['pad_shape']
        pad_h, pad_w = image_shape[0], image_shape[1]
        front_points = front_points.unsqueeze(1).repeat(1, V, 1, 1)

        #lidar_aug_matrix = torch.from_numpy(meta['lidar_aug_matrix']).float().to(front_points.device)
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(front_points.device)
        #lidars2imgs = torch.from_numpy(np.stack([meta['lidar2img'] for meta in img_metas])).float().to(front_points.device)
        #img_aug_matrix = torch.from_numpy(meta['img_aug_matrix']).float().to(front_points.device)
        #lidars2imgs_aug = torch.linalg.inv(lidar_aug_matrix) @ lidars2imgs @ img_aug_matrix[]


        proj_points = torch.einsum('bvnd, bvcd -> bvnc', torch.cat([front_points, front_points.new_ones(*front_points.shape[:-1], 1)], dim=-1), lidars2imgs)
        proj_points =  proj_points.view( B*V, N, -1 )
        #proj_points_clone = proj_points.clone()
        z_mask = (proj_points[..., 2:3] > 0) & ( proj_points[..., 2:3] < 1000.0 )
        proj_points_clone = proj_points[..., :3] / (proj_points[..., 2:3] + z_mask * 1e-6 - (~z_mask) * 1e-6)
        w_idx = torch.div( proj_points_clone[..., 0]*W, pad_w, rounding_mode='floor').long()
        h_idx = torch.div( proj_points_clone[..., 1]*H, pad_h, rounding_mode='floor').long()

        mask = ( w_idx < W ) & ( w_idx >= 0 ) & ( h_idx < H ) & ( h_idx >= 0 )
        mask &= z_mask.squeeze(-1)
        depth = proj_points[..., 2] 
        w_idx[~mask], h_idx[~mask], depth[~mask] = W, H, 1000.0
        ranks = h_idx*W + w_idx
        sort = ( ranks + depth/1001. ).argsort(dim = -1)
        
        w_idx = w_idx.gather(index = sort, dim = -1)
        h_idx = h_idx.gather(index = sort, dim = -1)
        depth = depth.gather(index = sort, dim = -1)
        ranks = ranks.gather(index = sort, dim = -1)
        front_points = front_points.view(B*V, N, 3)
        front_points = front_points.gather(index = sort.unsqueeze(-1).expand(-1, -1, 3), dim = 1 )
        kept = ranks.new_ones(*ranks.shape, dtype=torch.bool)
        kept[..., 1:] = (ranks[..., 1:] != ranks[..., :-1])

        #print( front_points.shape, kept.shape, w_idx.shape, h_idx.shape )

        # reference_points = img_feats.new_zeros( B*V, H, W, 3 )
        # reference_points_mask = img_feats.new_zeros( B*V, H, W, dtype=torch.bool )

        # for i in range(B*V):
        #     w_idx_i, h_idx_i =  w_idx[i, kept[i]], h_idx[i, kept[i]]
        #     front_points_i = front_points[i, kept[i], :]
        #     reference_points[i, h_idx_i[:-1], w_idx_i[:-1]] = front_points_i[:-1]
        #     reference_points_mask[i, h_idx_i[:-1], w_idx_i[:-1]] = True
            
        # reference_points = reference_points.view(  B, V, H, W, 3 )
        # reference_points_mask = reference_points_mask.view(  B, V, H, W )
        
        reference_points2 = img_feats.new_zeros( B*V, H*W+1, 3 )
        reference_points_mask2 = img_feats.new_zeros( B*V, H*W+1, dtype=torch.bool )
        
        #front_points_kept = torch.where( kept.unsqueeze(-1), front_points, front_points.new_zeros(1, 3) )
        ranks = torch.clamp( ranks, max = H*W )
        ranks = torch.where( kept, ranks, ranks.new_ones(1)*H*W )
        reference_points2.scatter_( 1, ranks.unsqueeze(-1).expand(-1, -1, 3), front_points )
        reference_points_mask2.scatter_( 1, ranks, kept )
        
        reference_points2 = reference_points2[:, :-1, :].contiguous().view( B, V,  H, W, -1 )
        reference_points_mask2 = reference_points_mask2[:, :-1].contiguous().view( B, V,  H, W )
        
        return reference_points2, reference_points_mask2


    def compute_frontpoints_corresponding_img_feat( self, front_points, img_feats, img_metas ):
        _, N, _ = front_points.shape
        B, V, C, H, W = img_feats.shape
        image_shape = img_metas[0]['pad_shape']
        pad_h, pad_w = image_shape[0], image_shape[1]
        front_points = front_points.unsqueeze(1).repeat(1, V, 1, 1)

        #lidar_aug_matrix = torch.from_numpy(meta['lidar_aug_matrix']).float().to(front_points.device)
        lidars2imgs = np.stack([meta['lidar2img'] for meta in img_metas])
        lidars2imgs = torch.from_numpy(lidars2imgs).float().to(front_points.device)
        #lidars2imgs = torch.from_numpy(np.stack([meta['lidar2img'] for meta in img_metas])).float().to(front_points.device)
        #img_aug_matrix = torch.from_numpy(meta['img_aug_matrix']).float().to(front_points.device)
        #lidars2imgs_aug = torch.linalg.inv(lidar_aug_matrix) @ lidars2imgs @ img_aug_matrix[]


        proj_points = torch.einsum('bvnd, bvcd -> bvnc', torch.cat([front_points, front_points.new_ones(*front_points.shape[:-1], 1)], dim=-1), lidars2imgs)
        z_mask = ( proj_points[..., 2:3] > 0 ) & ( proj_points[..., 2:3] < 1000.0 )
        proj_points_clone = proj_points[..., :3] / (proj_points[..., 2:3] + z_mask * 1e-6 - (~z_mask) * 1e-6)
        w_idx = torch.div( proj_points_clone[..., 0]*W, pad_w, rounding_mode='floor').long()
        h_idx = torch.div( proj_points_clone[..., 1]*H, pad_h, rounding_mode='floor').long() 
        img_idx = h_idx*W + w_idx     
          
        mask = ( w_idx < W ) & ( w_idx >= 0 ) & ( h_idx < H ) & ( h_idx >= 0 )
        mask &= z_mask.squeeze(-1)
        img_idx[~mask] = H*W   
        img_feats_flatten = img_feats.view( B, V, C, H*W ) 
        img_feats_flatten = torch.cat( (img_feats_flatten, img_feats_flatten.new_zeros( B, V, C, 1 )), dim = -1 )
        frontpoints_img_feat = img_feats_flatten.gather( dim = -1, index = img_idx.unsqueeze(-2).expand(-1, -1, C, -1) )
        frontpoints_aggregate_img_feat = frontpoints_img_feat.mean( dim = 1 )
        #print( frontpoints_aggregate_img_feat.shape )
        
        return frontpoints_aggregate_img_feat
           
        


    def padding_points_set_to_same_length(self, points_set: List[torch.Tensor],
                        padding_point: torch.Tensor = torch.tensor([0.0, 0.0, 0.0]),
                        padding_length = None):
        #B = len( points_set )
        if padding_length is None:
            padding_length = max( [points.shape[0] for points in points_set] )
        #max_len = max( [points.shape[0] for points in points_set] )
        points_set_padded = [torch.cat((points, padding_point.unsqueeze(0).repeat(padding_length - points.shape[0], 1).to(points.device)), dim = 0) for points in points_set]
        points_set_padded = torch.stack( points_set_padded, dim = 0 )
        
        return points_set_padded
    
    def query_denoising(self, feat_pc_flatten, bev_pos, batch_data_samples ):
        
        batch_gt_bboxes = [data_sample.gt_instances_3d.bboxes_3d.tensor.to(feat_pc_flatten.device)
                            for data_sample in batch_data_samples]
        batch_gt_labels = [data_sample.gt_instances_3d.labels_3d for data_sample in batch_data_samples]
        gt_nums = [gt_bboxes.size(0) for gt_bboxes in batch_gt_bboxes]
        
        
        mask_size = max( gt_nums )
        #gt_mask = torch.zeros( len(batch_data_samples), mask_size, mask_size, dtype=torch.bool ).to(feat_pc_flatten.device)
        denoised_points = torch.zeros( len(batch_data_samples), mask_size, 2 ).to(feat_pc_flatten.device)
        denoised_labels = torch.full( (len(batch_data_samples), mask_size), self.num_classes, dtype=torch.long ).to(feat_pc_flatten.device)
        for i, gt_bboxes in enumerate(batch_gt_bboxes):
            gt_centers, gt_box_scales = gt_bboxes[:, :3],  gt_bboxes[:, 3:6]
            gt_centers[:, 2] += gt_box_scales[:, 2] / 2
            if self.bbox_noise_scale > 0:
                noises = gt_box_scales / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(gt_centers) * 2 - 1.0
                gt_centers += torch.mul(rand_prob,
                                            noises) * self.bbox_noise_scale
            #torch.cat( ( query_reference_points[i], gt_centers ), dim = 0 ) 
            gt_num = gt_bboxes.size(0)
            #gt_mask[i, :gt_num, :gt_num] = True 
            
            denoised_points[i, :gt_num, :] = gt_centers[:, :2]
            denoised_labels[i, :gt_num] = batch_gt_labels[i].to(feat_pc_flatten.device)
            
        # add category embedding
        one_hot = F.one_hot(
            denoised_labels,
            num_classes=self.num_classes + 1)[..., :-1].contiguous().permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        
        bev_size = torch.div( denoised_points.new_tensor(self.train_cfg['grid_size'][:2]), self.train_cfg['out_size_factor'], rounding_mode='floor').long()
        voxel_size = denoised_points.new_tensor(self.train_cfg['voxel_size'][:2])
        denoised_bev_idx = torch.div( denoised_points - denoised_points.new_tensor(self.pc_range[:2]), 
            voxel_size * self.train_cfg['out_size_factor'],
            rounding_mode='floor').long()
        denoised_bev_idx[..., 0] = torch.clamp( denoised_bev_idx[..., 0], min = 0, max = bev_size[0] - 1 )
        denoised_bev_idx[..., 1] = torch.clamp( denoised_bev_idx[..., 1], min = 0, max = bev_size[1] - 1 )
        denoised_idx = denoised_bev_idx[..., 1] * bev_size[0] + denoised_bev_idx[..., 0]
        
        denoised_query_feat = feat_pc_flatten.gather( dim = -1, index = denoised_idx.unsqueeze(1).expand(-1, feat_pc_flatten.shape[1], -1) )
        
        denoised_query_embed = ( denoised_query_feat + query_cat_encoding ).transpose( 1, 2 ).contiguous()
        denoised_query_pos = bev_pos.gather( dim = -2, index = denoised_idx.unsqueeze(-1).expand(-1, -1, 2) )
        denoised_query_pos_embed = self.bev_embedding( self.pos2embed( denoised_query_pos, num_pos_feats=self.pos_embed_channel ) ) 
        
        return denoised_query_embed, denoised_query_pos, denoised_query_pos_embed, mask_size
    
    def gt_cenetrs_chosen(self, batch_data_samples, device):

        if self.training:
            batch_gt_centers = [data_sample.gt_instances_3d.bboxes_3d.gravity_center.to(device)
                                for data_sample in batch_data_samples]
        else:
            batch_gt_centers = [data_sample.eval_ann_info['gt_bboxes_3d'].gravity_center.to(device)
                                for data_sample in batch_data_samples] 
        gt_centers = self.padding_points_set_to_same_length( batch_gt_centers, padding_length= self.num_proposals )           
        return gt_centers       
    
    def gt_frontpoints_chosen(self, points_set, batch_data_samples):
        frontpoints = []
        for i, data_sample in enumerate( batch_data_samples ):
            points = points_set[i]
            gt_box_tensor = data_sample.gt_instances_3d.bboxes_3d.tensor.to( points.device )
            gt_box_3d = BaseInstance3DBoxes(gt_box_tensor[:,:7])
            points_inbox_matrix = gt_box_3d.points_in_boxes_all(points)
            frontpoint = points[points_inbox_matrix.any(dim=1)]
            frontpoints.append( frontpoint )
            
        frontpoints = self.padding_points_set_to_same_length( frontpoints )
        return frontpoints  
    
    def gt_fronttokenpoints_chosen(self, points_set, batch_data_samples):
        
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        out_size_factor = self.train_cfg['out_size_factor']
        batch_size =  len( batch_data_samples )
        heatmap = points_set.new_zeros( batch_size, grid_size[0] // out_size_factor, grid_size[1] //out_size_factor, dtype=torch.bool  )
        for i, data_sample in enumerate( batch_data_samples ):
            points = points_set[i]
            gt_bboxes_3d = data_sample.gt_instances_3d.bboxes_3d.tensor.to( points.device )
            #print(gt_bboxes_3d.tensor.shape )
            for idx in range(len(gt_bboxes_3d)):
                width = gt_bboxes_3d[idx][3]
                length = gt_bboxes_3d[idx][4]
                width = width / voxel_size[0] / self.train_cfg['out_size_factor']
                length = length / voxel_size[1] / self.train_cfg['out_size_factor']  
                if width > 0 and length > 0:    
                    # radius_heatmap = int(gaussian_radius(
                    #     (length, width),
                    #     min_overlap=self.train_cfg['gaussian_overlap']))+1
                    #radius_heatmap = int(math.sqrt( length*length + width*width))+1
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                    coor_x = ((x - pc_range[0]) / voxel_size[0] /
                                self.train_cfg['out_size_factor'])
                    coor_y = ((y - pc_range[1]) / voxel_size[1] /
                                self.train_cfg['out_size_factor'])

                    center = points_set.new_tensor( [coor_x, coor_y] ).long()
                    #draw_bool_heatmap( heatmap[i], center[[1, 0]], radius_heatmap )
                    heatmap[i, center[0], center[1]] = True
                    
        front_points_mask = self.get_front_points_mask( points_set, heatmap )
        frontpoints =  [ points_set[i, front_points_mask[i]]  for i in range(batch_size) ]
        frontpoints = self.padding_points_set_to_same_length( frontpoints )
        return frontpoints
       
       
    def frontpoints_encoding( self, frontpoints, feats_dict ):
        
        frontpoints_dict = dict( points=frontpoints )
        return self.points_encoder( frontpoints_dict, feats_dict, None )
    
    def compute_seed_feature(self, seed_points, voxel_feats_dict ):
        
        normalized_points = 2*(seed_points - seed_points.new_tensor([0.0, 0.0, -1.0])) / (seed_points.new_tensor(self.pc_range[3:]) - seed_points.new_tensor(self.pc_range[:3])) 
        #normalized_points = torch.stack( (normalized_points[...,2], normalized_points[...,1], normalized_points[...,0] ), dim=-1)
        seed_voxel_feats = []
        for i, voxel_feat in enumerate(voxel_feats_dict['multi_scale_3d_feats']):
            if i < 2:
                continue
            voxel_feat = voxel_feat.dense().float()
            seed_voxel_feat = F.grid_sample( voxel_feat, #copy.deepcopy(voxel_feats.detach()), 
                            normalized_points.unsqueeze(1).unsqueeze(1),
                            mode='bilinear',
                            padding_mode='border',
                            align_corners=False).squeeze(2).squeeze(2) 
            seed_voxel_feats.append( seed_voxel_feat )
            
        seed_voxel_feat = torch.cat( seed_voxel_feats, dim=1 )
        

        pc_feat = voxel_feats_dict['spatial_feats']
        seed_pillar_feat = F.grid_sample( pc_feat.float(),  #copy.deepcopy(feat_pc.detach())
                                  normalized_points[..., :2].unsqueeze(1),
                                  mode='bilinear',
                                  padding_mode='border',
                                  align_corners=False).squeeze(2)
        seed_feat = torch.cat( (seed_voxel_feat, seed_pillar_feat), dim=1 ) 
        seed_feat  = self.seed_embedding( seed_feat )
        
        return seed_feat 
    
    @torch.no_grad()  
    def get_front_bev_points( self, points, front_idx, front_token_number ):
        # points: [B, N, 3 ]
        # front_idx: [B, num_proposals]
        B, N, _ = points.shape
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        bev_size = torch.div(points.new_tensor(cfg['grid_size'][:2]), cfg['out_size_factor'], rounding_mode='floor').long()
        voxel_size = points.new_tensor(cfg['voxel_size'])
        bev_xy = (points[..., :2]-points.new_tensor(self.pc_range[:2])) / voxel_size[:2] / cfg['out_size_factor']
        z = points[..., 2] - points.new_tensor(self.pc_range[5])
        bev_idx = torch.floor(bev_xy).long()
        # bev_idx = torch.div((points[..., :2]-points.new_tensor(self.pc_range[:2])), 
        #     voxel_size[:2] * cfg['out_size_factor'],
        #       rounding_mode='floor').long()
        assert  ( bev_idx >= 0).all() & (bev_idx <= bev_size).all() 
        bev_idx[..., 0] = torch.clamp( bev_idx[..., 0], min = 0, max = bev_size[0] - 1 )
        bev_idx[..., 1] = torch.clamp( bev_idx[..., 1], min = 0, max = bev_size[1] - 1 )
        idx = bev_idx[..., 1] * bev_size[0] + bev_idx[..., 0]
        front_points_mask = ( idx.unsqueeze(-1) == front_idx.unsqueeze(-2) ).any(dim = -1)
        #print( front_points_mask.shape )
        
        front_bev_idx = idx.new_zeros(*idx.shape)
        for i in range(B):
            idx_row, idx_col = torch.nonzero( front_idx[i].unsqueeze(0) == idx[i].unsqueeze(1), as_tuple=True )
            front_bev_idx[i, idx_row] = idx_col
        front_bev_idx = torch.where( front_points_mask, front_bev_idx, front_token_number * torch.ones_like(front_bev_idx) )

        
        weights_sum = z.new_zeros(B, front_token_number + 1 ) 
        weights_sum.scatter_add_( 1, front_bev_idx, torch.exp(z) ) 
        #print( weights_sum[0] )
        weights_sum = weights_sum.gather( 1, front_bev_idx )
        normalized_weights = torch.exp(z) / weights_sum
    

        front_bev_dix = bev_xy.new_zeros( B, front_token_number + 1, 2 )
        front_bev_dix.scatter_add_( 1, front_bev_idx.unsqueeze(-1).expand(-1, -1, 2), bev_xy * normalized_weights.unsqueeze(-1))                

        return front_bev_dix[:, :-1].contiguous()       
    
    def forward_single(self, points, voxel_feats_dict, img_feats, batch_data_samples):
        
        self.num_proposals = self.num_proposals_ori
        #print("3:", len(voxel_feats_dict['spatial_feats']) )
        #spatial_feat = voxel_feats_dict['spatial_feats']
        pillar_feat = voxel_feats_dict['pillar_feats']
        
        batch_size, _, H, W = pillar_feat.shape
        feat_pc = self.shared_conv(pillar_feat)
        
        multiscale_inputs = [feat_pc]
        if self.multiscale:
            multiscale_inputs.append( self.dconv(multiscale_inputs[-1]) )
            multiscale_inputs.append( self.dconv2(multiscale_inputs[-1]) )
        multiscale_inputs_flatten = torch.cat([i.flatten(2,3) for i in multiscale_inputs], dim=-1)

        #voxel_feats_dict['spatial_feats'] = feat_pc
        
        cfg = self.train_cfg if self.train_cfg else self.test_cfg
        voxel_size = feat_pc.new_tensor(cfg['voxel_size'])   
        # bev_size = torch.div(feat_pc.new_tensor(cfg['grid_size'][:2]), cfg['out_size_factor'], rounding_mode='floor')

        start_points = feat_pc.new_tensor(cfg['point_cloud_range'][:3])
        points = self.padding_points_set_to_same_length( points, start_points )
        
        #################################
        # image to BEV
        #################################
        feat_pc_flatten = feat_pc.view(batch_size,
                                    feat_pc.shape[1],
                                    -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(feat_pc_flatten.device)
        
        with torch.autocast('cuda', enabled=False):
            dense_heatmap = self.heatmap_head( feat_pc.float() )    #copy.deepcopy(feat_pc.detach())

        heatmap = dense_heatmap.detach().sigmoid()  #.view(batch_size, -1)
        #frontmap, _ = heatmap.max(dim=1)
        #print( frontmap.shape )
        
        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding),
                padding:(-padding)] = local_max_inner
        # for Pedestrian & Traffic_cone in nuScenes
        if self.train_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(
                heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(
                heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.train_cfg[
                'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
            local_max[:, 1, ] = F.max_pool2d(
                heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(
                heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
        
        top_proposals = heatmap.view(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='floor')
        top_proposals_index = top_proposals % heatmap.shape[-1]
        
        self.query_labels = top_proposals_class
        
        # add category embedding
        one_hot = F.one_hot(
            top_proposals_class,
            num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())

        query_pos = bev_pos.gather( dim = 1, index = top_proposals_index.unsqueeze(-1).expand(-1, -1, 2) )

        query_pos_embed = self.bev_embedding( self.pos2embed(query_pos, num_pos_feats=self.pos_embed_channel ) )
        #print( query_pos_embed.shape )
        front_feature = feat_pc_flatten.gather( dim = -1, index = top_proposals_index.unsqueeze(1).expand(-1, feat_pc_flatten.shape[1], -1) )
        query_embed = ( front_feature + query_cat_encoding ).transpose( 1, 2 ).contiguous()
        
        query_heatmap_score = heatmap.gather( index=top_proposals_index[:,None, :].expand(-1, self.num_classes,-1), dim=-1 ) 
    
        if self.multistage_heatmap != 0:
            query_labels = [top_proposals_class]
            query_embeds = [query_embed]
            query_poses = [query_pos]
            query_pos_embeds = [query_pos_embed]
            query_heatmap_scores = [query_heatmap_score]
            acc_masks = torch.ones_like(dense_heatmap, dtype=torch.bool)
            multistage_masks = [acc_masks]
            heatmap_train = [dense_heatmap]
            for i in range(self.multistage_heatmap):
                
                selected_mask = torch.zeros_like(acc_masks, dtype=torch.bool).view(batch_size, -1)
                selected_mask = torch.scatter( selected_mask, 1, top_proposals, 1 )
                selected_mask = selected_mask.reshape(*acc_masks.shape)

                dense_heatmap = self.multistage_heatmap_head[i](feat_pc.float())
                heatmap = dense_heatmap.detach().sigmoid()
                heatmap_train.append( dense_heatmap )

                padding = self.nms_kernel_size // 2
                local_max = torch.zeros_like(heatmap)
                # equals to nms radius = voxel_size * out_size_factor * kenel_size
                local_max_inner = F.max_pool2d(
                    heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
                local_max[:, :, padding:(-padding),
                        padding:(-padding)] = local_max_inner
                # for Pedestrian & Traffic_cone in nuScenes
                if self.train_cfg['dataset'] == 'nuScenes':
                    local_max[:, 8, ] = F.max_pool2d(
                        heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                    local_max[:, 9, ] = F.max_pool2d(
                        heatmap[:, 9], kernel_size=1, stride=1, padding=0)
                    # selected_mask[:, :8] = F.interpolate( selected_mask[:, :8].float(), size = dense_heatmap.shape[-2:], mode='nearest' )
                elif self.train_cfg[
                        'dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                    local_max[:, 1, ] = F.max_pool2d(
                        heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                    local_max[:, 2, ] = F.max_pool2d(
                        heatmap[:, 2], kernel_size=1, stride=1, padding=0)
                    # selected_mask[:, 0] = F.interpolate( selected_mask[:, 0].float(), size = dense_heatmap.shape[-2:], mode='nearest' )
                    # selected_mask[:, 3:] = F.interpolate( selected_mask[:, 3:].float(), size = dense_heatmap.shape[-2:], mode='nearest' )
                
                heatmap = heatmap * (heatmap == local_max)
                acc_masks = acc_masks * (~selected_mask)
                heatmap = heatmap * acc_masks
                heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)
                        
                top_proposals = heatmap.view(batch_size, -1).argsort(
                    dim=-1, descending=True)[..., :self.num_proposals]
                top_proposals_class = torch.div(top_proposals, heatmap.shape[-1], rounding_mode='floor')
                top_proposals_index = top_proposals % heatmap.shape[-1]
                
                query_labels.append( top_proposals_class )
                
                # add category embedding
                one_hot = F.one_hot(
                    top_proposals_class,
                    num_classes=self.num_classes).permute(0, 2, 1)
                query_cat_encoding = self.class_encoding(one_hot.float())

                query_pos = bev_pos.gather( dim = 1, index = top_proposals_index.unsqueeze(-1).expand(-1, -1, 2) )

                query_pos_embed = self.bev_embedding( self.pos2embed(query_pos, num_pos_feats=self.pos_embed_channel ) )
                #print( query_pos_embed.shape )
                front_feature = feat_pc_flatten.gather( dim = -1, index = top_proposals_index.unsqueeze(1).expand(-1, feat_pc_flatten.shape[1], -1) )
                query_embed = ( front_feature + query_cat_encoding ).transpose( 1, 2 ).contiguous()

                query_heatmap_score = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)

                query_embeds.append(query_embed)
                query_poses.append(query_pos)
                query_pos_embeds.append(query_pos_embed)
                query_heatmap_scores.append(query_heatmap_score)
                                
                multistage_masks.append(acc_masks.clone())
            
            self.query_label = torch.cat( query_labels, dim = 1 )   
            query_embed = torch.cat(query_embeds, dim=1)
            #print( query_embed.shape )
            query_pos = torch.cat(query_poses, dim=1)
            query_pos_embed = torch.cat(query_pos_embeds, dim=1)
            query_heatmap_score = torch.cat(query_heatmap_scores, dim=2)
            
            self.num_proposals = self.num_proposals_ori * ( self.multistage_heatmap + 1 )
            
              
        
        # query denoising
        if self.training:
            denoised_query_embed, denoised_query_pos, denoised_query_pos_embed, mask_size = self.query_denoising( feat_pc_flatten, bev_pos, batch_data_samples )
            query_embed = torch.cat( (query_embed, denoised_query_embed), dim = 1 )
            query_pos = torch.cat( (query_pos, denoised_query_pos), dim = 1 )
            query_pos_embed = torch.cat( (query_pos_embed, denoised_query_pos_embed), dim = 1 )
            
            attn_mask = torch.zeros( self.num_proposals + mask_size, self.num_proposals + mask_size, dtype=torch.bool ).to(feat_pc_flatten.device)
            #attn_mask[:, self.num_proposals:, :self.num_proposals] = True
            attn_mask[:self.num_proposals, self.num_proposals:] = True
            attn_mask[self.num_proposals:, :self.num_proposals] = True
            
            self.num_proposals += mask_size
                
        ################## Deformable Parameters #############
        if not self.multiscale:
            #W, H = lidar_feat.shape[-2:]
            spatial_shapes = torch.as_tensor([ [W, H] ], dtype=torch.long, device='cuda')
            level_start_index = torch.as_tensor([0,], dtype=torch.long, device='cuda')
        else:
            spatial_shapes = torch.as_tensor([ i.shape[2:] for i in multiscale_inputs ], dtype=torch.long, device='cuda')
            level_start_index = torch.as_tensor([0, *(torch.cumsum(torch.prod(spatial_shapes, dim=1), dim=0)[:-1])], dtype=torch.long, device='cuda')
            
            # lidar feat
            #lidar_feat_flatten = multiscale_inputs_flatten
            # bev_pos2 = self.bev_pos2.repeat(batch_size, 1, 1).to(multiscale_inputs_flatten.device)
            # bev_pos4 = self.bev_pos4.repeat(batch_size, 1, 1).to(multiscale_inputs_flatten.device)
            # bev_pos = torch.cat([bev_pos, bev_pos2, bev_pos4], dim=1)        
        
        kwargs = dict(
            spatial_shapes = spatial_shapes,
            level_start_index = level_start_index,
            valid_ratios = torch.ones((batch_size, 1, 2), device='cuda'),
        )       
        
        #print(spatial_shapes)
        reference_points = query_pos / spatial_shapes[:1].unsqueeze(0).to(query_pos.device)
        reference_points = reference_points.unsqueeze(2)
        query_embed = self.center_transformer_decoder(             
            query_embed,
            query_pos= query_pos_embed,
            value= multiscale_inputs_flatten.transpose( 1, 2 ).contiguous(),
            reference_points = reference_points,
            self_attn_mask = attn_mask,
            **kwargs       
        )
        query_embed = torch.nan_to_num( query_embed )
    
        center_dict = self.center_head(query_embed.transpose( 1, 2 ).contiguous())  

        query_pos += center_dict['center'].permute( 0, 2, 1 )   
        query_pos_embed = self.bev_embedding( self.pos2embed( query_pos, num_pos_feats=self.pos_embed_channel ))
        
        
        cls_key_embed = feat_pc_flatten.transpose( 1, 2 )
        cls_key_pos_embed = self.bev_embedding(self.pos2embed( bev_pos, num_pos_feats = self.pos_embed_channel ))
        
        box_embed, cls_embed = self.decoder( 
            query_embed,
            query_pos = query_pos_embed,
            box_value = multiscale_inputs_flatten.transpose( 1, 2 ),
            reference_points = reference_points,
            cls_key= cls_key_embed,
            cls_key_pos= cls_key_pos_embed,
            self_attn_mask = attn_mask,
            **kwargs  
            )
        box_embed = torch.nan_to_num( box_embed )
        cls_embed = torch.nan_to_num( cls_embed )

        # Prediction
        ret_dict = self.prediction_head(box_embed.transpose( 1, 2 ).contiguous(),
                                        cls_embed.transpose( 1, 2 ).contiguous(),
                                        )        
        ret_dict['center'] = query_pos.permute( 0, 2, 1 )

        ret_dict['dense_heatmap'] = dense_heatmap
        if self.multistage_heatmap != 0:
            ret_dict['multistage_masks'] = multistage_masks
            ret_dict['dense_heatmap'] = heatmap_train
            
        ret_dict['query_heatmap_score'] = query_heatmap_score
        # if img_feats is not None:
        #     ret_dict['frontscore'] = frontscore
        #     ret_dict['front_bev_map'] = front_bev_mask.view( batch_size, H, W )
       
        return [ret_dict]
        
    
    def forward(self, points, voxel_feats_dict, img_feats, batch_data_samples):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        #if isinstance(points, torch.Tensor):
        #    points = [points]
        # if isinstance(pillar_feat, torch.Tensor):
        #     pillar_feat = [pillar_feat]
        # if isinstance(voxel_feats, torch.Tensor):
        #     voxel_feats = [voxel_feats]
        if isinstance(img_feats, torch.Tensor):
            img_feats = [img_feats]
        res = multi_apply(self.forward_single, [points], [voxel_feats_dict], img_feats, [batch_data_samples])
        assert len(res) == 1, 'only support one level features.'
        return res
    
    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: dict):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (dict): The prediction results. The dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key in preds_dict.keys():
                pred_dict[key] = preds_dict[key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)


        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
        )
        
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0) 
        # iou_bbox_target = torch.cat(res_tuple[8], dim=0)
        # iou_bbox_weights = torch.cat(res_tuple[9], dim=0)  
        # iou3d_dec = torch.cat(res_tuple[10], dim=0)
        # frontmap = torch.cat(res_tuple[8], dim=0)
        # front_recall = np.mean(res_tuple[9])     
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            # iou_bbox_target,
            # iou_bbox_weights,
            # iou3d_dec,
            # frontmap,
            # front_recall,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including 
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        score = torch.nan_to_num(score)
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        bboxes_tensor = torch.nan_to_num(bboxes_tensor)
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        #print( gt_bboxes_tensor.shape )


        assign_result = None
        if self.train_cfg.assigner.type == 'HungarianAssigner3D':
            assign_result = self.bbox_assigner.assign(
                bboxes_tensor,
                gt_bboxes_tensor,
                gt_labels_3d,
                score,
                self.train_cfg,
            )
        elif self.train_cfg.assigner.type == 'HeuristicAssigner':
            assign_result = self.bbox_assigner.assign(
                bboxes_tensor,
                gt_bboxes_tensor,
                None,
                gt_labels_3d,
                #self.query_labels[batch_idx],
            )
        else:
            raise NotImplementedError

        # 1. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 2. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        
        feature_map_size = (torch.div(grid_size[:2], self.train_cfg['out_size_factor'],
                            rounding_mode='floor'))  # [x_len, y_len]


        heatmap = gt_bboxes_3d.new_zeros( self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        #frontmap =  gt_bboxes_3d.new_zeros( feature_map_size[1], feature_map_size[0], dtype=torch.bool)
        for idx in range(gt_labels_3d.shape[0]):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']  
            if width > 0 and length > 0:   
                radius_heatmap = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius_heatmap = max(self.train_cfg['min_radius'], int(radius_heatmap)) 
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                            self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                            self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                        dtype=torch.float32,
                                        device=device)
                center_int = center.to( torch.int32 )  
                
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                    center_int, radius_heatmap)
                #radius_frontmap = int(math.sqrt( length*length + width*width))+1
                #draw_bool_heatmap( frontmap, center_int, radius_frontmap )   #[gt_labels_3d[idx]]
                # if center_int[0] < feature_map_size[1] and center_int[1] < feature_map_size[0]:
                #     frontmap[center_int[1], center_int[0]] = True    
            else:
                print("#### impossible ####")
        
        # front_bev_map = copy.deepcopy(preds_dict['front_bev_map'].detach()).squeeze(0)   
        # front_recall = torch.sum( frontmap & front_bev_map ).float() / torch.sum( frontmap ).float()
        
        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        
        # iou_bbox_target = torch.zeros([num_proposals, gt_bboxes_tensor.shape[-1]
        #                     ]).to(center.device)
        # iou_bbox_weights = torch.zeros([num_proposals, gt_bboxes_tensor.shape[-1]
        #                             ]).to(center.device)
        # iou_bbox_target[pos_inds, :] = sampling_result.pos_gt_bboxes
        # iou_bbox_weights[pos_inds, :] = 1.0
        # iou2d = bbox_overlaps_nearest_3d( bboxes_tensor, iou_bbox_target, is_aligned=True, coordinate='depth' )
        # z1, z2, z3, z4 = self._bbox_to_loss(bboxes_tensor)[:, 2], self._bbox_to_loss(bboxes_tensor)[:, 5], self._bbox_to_loss(iou_bbox_target)[:, 2], self._bbox_to_loss(iou_bbox_target)[:, 5]
        # iou_z = torch.max(torch.min(z2, z4) - torch.max(z1, z3), z1.new_zeros(z1.shape)) / (torch.max(z2, z4) - torch.min(z1, z3) )
        # iou3d_dec = (iou2d + iou_z)/2
        
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
            # iou_bbox_target[None],
            # iou_bbox_weights[None],
            # iou3d_dec[None],
            # frontmap[None],
            # float(front_recall),
        )
    
    def loss(self, points, voxel_feats_dict, img_feats, batch_data_samples):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        if points is None:
            points = [None]
        # pillar_feat = voxel_feats_dict.get('spatial_feats', None)
        # voxel_feats = voxel_feats_dict.get('multi_scale_3d_feats', None)
        # if pillar_feat is None:
        #     pillar_feat = [None]
        # if voxel_feats is None:
        #     voxel_feats = [None]
        # else:
        #     voxel_feats = voxel_feats[-1]
        if img_feats is None:
            img_feats = [None]
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(points, voxel_feats_dict, img_feats, batch_data_samples )
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
            # iou_bbox_target,
            # iou_bbox_weights,
            # iou3d_dec,
            # frontmap,
            # front_recall,

        ) = self.get_targets( batch_gt_instances_3d, preds_dicts[0][0] )
        if hasattr(self, 'on_the_image_mask'):
            label_weights = label_weights * self.on_the_image_mask
            bbox_weights = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()
        loss_dict = dict()
        preds_dict = preds_dicts[0][0]


        # compute heatmap loss
        if self.multistage_heatmap != 0:
            multistage_masks = torch.cat(preds_dict['multistage_masks'], dim=0)
            heatmap = heatmap.repeat(len(preds_dict['dense_heatmap']), 1, 1, 1) * multistage_masks # mask the ignore gt
            preds_dict['dense_heatmap'] = torch.cat(preds_dict['dense_heatmap'], dim=0)
            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(preds_dict['dense_heatmap']), 
                heatmap.float(), 
                weight=multistage_masks, 
                avg_factor=max(heatmap.eq(1).float().sum().item(), 1)
            )
        else:
            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(preds_dict['dense_heatmap']).float(),
                heatmap.float(),
                avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
            )
        loss_dict['loss_heatmap'] = loss_heatmap
        
        # loss_dict['loss_frontmap'] = self.loss_frontmap(
        #     preds_dict['frontscore'].float(),
        #     frontmap.float(),
        #     avg_factor=max(frontmap.eq(1).float().sum().item(), 1),
        # )
        # loss_dict['front_recall'] = loss_heatmap.new_tensor( front_recall ) 
        
        # compute loss
        layer_score = preds_dict['heatmap']
        cls_score = layer_score.permute(0, 2, 1).reshape(
            -1, self.num_classes)
        loss_cls = self.loss_cls(
            cls_score.float(),
            labels.reshape(-1),
            label_weights.reshape(-1),
            avg_factor=max(num_pos, 1),
        )
        # loss_cls = self.loss_cls(
        #     cls_score.float(),
        #     [labels.reshape(-1), iou3d_dec.reshape(-1)],
        #     label_weights.reshape(-1),
        #     avg_factor=max(num_pos, 1),
        # )

        center = preds_dict['center']
        height = preds_dict['height']
        dim = preds_dict['dim']
        rot = preds_dict['rot']
        preds = torch.cat(
            [center, height, dim, rot],
            dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
        
        if 'vel' in preds_dict.keys():
            vel = preds_dict['vel']
            preds = torch.cat([
                center, height, dim, rot, vel
            ], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            #pred_iou_box = self.bbox_coder.decode_only_box( rot, dim, center, height, vel)
        #else:
            #pred_iou_box = self.bbox_coder.decode_only_box( rot, dim, center, height)
            
        # compute bbox and class loss
        code_weights = self.train_cfg.get('code_weights', None)
        reg_weights = bbox_weights * bbox_weights.new_tensor(  # noqa: E501
            code_weights )
        loss_bbox = self.loss_bbox(
            preds,
            bbox_targets,
            reg_weights,
            avg_factor=max(num_pos, 1))
        
        loss_cls = torch.nan_to_num( loss_cls )
        loss_bbox = torch.nan_to_num( loss_bbox )
        loss_dict['cls_loss'] = loss_cls
        loss_dict['loss_bbox'] = loss_bbox
        loss_dict['matched_ious'] = loss_cls.new_tensor(matched_ious)
        
        # #iou_loss
        # pred_iou_box = pred_iou_box.reshape(-1, pred_iou_box.shape[-1])
        # iou_bbox_target = iou_bbox_target.reshape(-1, iou_bbox_target.shape[-1])
        # iou_bbox_weights = iou_bbox_weights.reshape(-1, iou_bbox_weights.shape[-1])
        # #print( pred_iou_box.shape, iou_bbox_target.shape, reg_weights.shape)
        # loss_iou = self.loss_iou( pred_iou_box,
        #                             iou_bbox_target,
        #                             iou_bbox_weights,
        #                             avg_factor=max(num_pos, 1)
        #                              )
        # z1, z2, z3, z4 = self._bbox_to_loss(pred_iou_box)[:, 2], self._bbox_to_loss(pred_iou_box)[:, 5], self._bbox_to_loss(iou_bbox_target)[:, 2], self._bbox_to_loss(iou_bbox_target)[:, 5]
        # iou_z = torch.max(torch.min(z2, z4) - torch.max(z1, z3), z1.new_zeros(z1.shape)) / (torch.max(z2, z4) - torch.min(z1, z3) )
        # loss_iou_z = 1 - iou_z
        # loss_iou += torch.sum(loss_iou_z * iou_bbox_weights[:, 0]) / max(num_pos, 1)
        # loss_dict['loss_iou'] = loss_iou
        #loss_dict['matched_ious'] = loss_cls.new_tensor(matched_ious)
        #loss_dict['decouple_ious'] = torch.mean( iou3d_dec )
        
        return loss_dict
    
    def predict(self, points, voxel_feats_dict, img_feats, batch_data_samples ):    
        if points is None:
            points = [None]
        # if pillar_feat is None:
        #     pillar_feat = [None]
        # if voxel_feats is None:
        #     voxel_feats = [None]
        if img_feats is None:
            img_feats = [None]
        
        
        batch_input_metas, batch_gt_instances_3d = [], []
        # point = points[0]
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
        #     #print( data_sample.eval_ann_info )
        #     bboxes_3d_valid = point.new_tensor( data_sample.eval_ann_info['bbox_3d_isvalid'], dtype=torch.bool )
        #     bboxes_3d = data_sample.eval_ann_info['gt_bboxes_3d'].tensor.to( point.device ) 
        #     labels_3d = point.new_tensor( data_sample.eval_ann_info['gt_labels_3d'], dtype=torch.long )
        #     bboxes_3d = bboxes_3d[bboxes_3d_valid]
        #     labels_3d = labels_3d[bboxes_3d_valid]
        #     bboxes_3d = LiDARInstance3DBoxes( bboxes_3d, box_dim = bboxes_3d.shape[-1] )
        #     gt_instances_3d = InstanceData( bboxes_3d = bboxes_3d,
        #                                 labels_3d = labels_3d )
            
        #     batch_gt_instances_3d.append(gt_instances_3d)
        preds_dicts = self(points, voxel_feats_dict, img_feats, batch_data_samples )
        res = self.predict_by_feat(preds_dicts, batch_input_metas, batch_gt_instances_3d )
        return res
    
    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        batch_gt_instances_3d,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """

        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()

            one_hot = F.one_hot(
               self.query_labels,
               num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[0]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()
                #temp_instances.l1_dist = anna_dict['l1_dist'].unsqueeze(0).repeat( temp_instances.bboxes_3d.tensor.shape[0], 1 )

                ret_layer.append(temp_instances)
                
                # metric = dict( l1_dist= l1_dist[i],
                #               l1_dist_x= l1_dist_x[i],
                #               l1_dist_y= l1_dist_y[i],
                #               l1_dist_z= l1_dist_z[i],
                #               detect_recall= detect_recall[i],
                #               front_recall= front_recall[i],
                #               )
                # metric = dict( front_recall= front_recall[i], )
                # additional_metrics.append(metric)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]  #, additional_metrics

    @staticmethod
    def _bbox_to_loss(bbox):
        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)