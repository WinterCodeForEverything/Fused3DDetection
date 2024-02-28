# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional, Tuple, Union
from mmdet.models import  DetrTransformerDecoderLayer
import torch
from torch import Tensor, nn

from mmdet3d.registry import MODELS

from mmengine.model import ModuleList, BaseModel, BaseModule

from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from .attention import FlashMHA

@MODELS.register_module()
class PETRMultiheadFlashAttention(BaseModel):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='Dropout', drop_prob=0.),
                 init_cfg=None,
                 batch_first=True,
                 **kwargs):
        super(PETRMultiheadFlashAttention, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn(
                'The arguments `dropout` in MultiheadAttention '
                'has been deprecated, now you can separately '
                'set `attn_drop`(float), proj_drop(float), '
                'and `dropout_layer`(dict) ', DeprecationWarning)
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = False

        self.attn = FlashMHA(embed_dims, num_heads, attn_drop, dtype=torch.float16, device='cuda',
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    # @deprecated_api_warning({'residual': 'identity'},
    #                         cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
            [num_queries, bs, embed_dims]
            if self.batch_first is False, else
            [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
        
        out = self.attn(
            q=query,
            k=key,
            v=value,
            key_padding_mask=None)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        return identity + self.dropout_layer(self.proj_drop(out))

class PositionEncodingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats), nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding
    
    

# @MODELS.register_module()
# class CenterTransformerDecoderLayer(DeformableDetrTransformerDecoderLayer):

#     def __init__(self,
#                  #pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
#                  **kwargs):
#         super().__init__(**kwargs)
        
#         #self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
#         #self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        
#     def _init_layers(self) -> None:
#         """Initialize self-attention, FFN, and normalization."""
#         self.self_attn = MultiheadAttention(**self.self_attn_cfg)
#         self.cross_attn = MultiheadAttention(**self.cross_attn_cfg)
#         self.embed_dims = self.self_attn.embed_dims
#         self.ffn = FFN(**self.ffn_cfg)
#         norms_list = [
#             build_norm_layer(self.norm_cfg, self.embed_dims)[1]
#             for _ in range(3)
#         ]
#         self.norms = ModuleList(norms_list)

#     def forward(self,
#                 query: Tensor,
#                 key: Tensor = None,
#                 value: Tensor = None,
#                 query_pos: Tensor = None,
#                 key_pos: Tensor = None,
#                 self_attn_mask: Tensor = None,
#                 cross_attn_mask: Tensor = None,
#                 key_padding_mask: Tensor = None,
#                 **kwargs) -> Tensor:
#         """
#         Args:
#             query (Tensor): The input query, has shape (bs, num_queries, dim).
#             key (Tensor, optional): The input key, has shape (bs, num_keys,
#                 dim). If `None`, the `query` will be used. Defaults to `None`.
#             value (Tensor, optional): The input value, has the same shape as
#                 `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
#                 `key` will be used. Defaults to `None`.
#             query_pos (Tensor, optional): The positional encoding for `query`,
#                 has the same shape as `query`. If not `None`, it will be added
#                 to `query` before forward function. Defaults to `None`.
#             key_pos (Tensor, optional): The positional encoding for `key`, has
#                 the same shape as `key`. If not `None`, it will be added to
#                 `key` before forward function. If None, and `query_pos` has the
#                 same shape as `key`, then `query_pos` will be used for
#                 `key_pos`. Defaults to None.
#             self_attn_mask (Tensor, optional): ByteTensor mask, has shape
#                 (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
#                 (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
#                 Defaults to None.
#             key_padding_mask (Tensor, optional): The `key_padding_mask` of
#                 `self_attn` input. ByteTensor, has shape (bs, num_value).
#                 Defaults to None.

#         Returns:
#             Tensor: forwarded results, has shape (bs, num_queries, dim).
#         """
#         query = self.self_attn(
#             query=query,
#             key=query,
#             value=query,    # + query_pos
#             query_pos=query_pos,
#             key_pos=query_pos,
#             attn_mask=self_attn_mask,
#             **kwargs)
#         query = self.norms[0](query)
#         # Note that the `value` (equal to `key`) is encoded with `key_pos`.
#         # This is different from the standard DETR Decoder Layer.
#         #with torch.autocast('cuda'):
#         query = self.cross_attn(
#             query=query,
#             key=key,
#             value=key, #+ key_pos
#             query_pos=query_pos,
#             key_pos=key_pos,
#             attn_mask=cross_attn_mask,
#             key_padding_mask=key_padding_mask,
#             **kwargs)
#         query = self.norms[1](query)
#         query = self.ffn(query)
#         query = self.norms[2](query)

#         #query = query.transpose(1, 2)
#         return query




@MODELS.register_module()
class MFDeformableDetrTransformerDecoderLayer(BaseModule):
    def __init__(self,
                    self_attn_cfg,
                    cls_cross_attn_cfg,
                    box_cross_attn_cfg,
                    ffn_cfg,
                    norm_cfg = dict(type='LN'),
                    init_cfg = None) -> None:

        super().__init__(init_cfg=init_cfg)

        self.self_attn_cfg = self_attn_cfg
        self.box_cross_attn_cfg = box_cross_attn_cfg
        self.cls_cross_attn_cfg = cls_cross_attn_cfg
        if 'batch_first' not in self.self_attn_cfg:
            self.self_attn_cfg['batch_first'] = True
        else:
            assert self.self_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        if 'batch_first' not in self.cls_cross_attn_cfg:
            self.cls_cross_attn_cfg['batch_first'] = True
        else:
            assert self.cls_cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'
            
        if 'batch_first' not in self.box_cross_attn_cfg:
            self.box_cross_attn_cfg['batch_first'] = True
        else:
            assert self.box_cross_attn_cfg['batch_first'] is True, 'First \
            dimension of all DETRs in mmdet is `batch`, \
            please set `batch_first` flag.'

        self.ffn_cfg = ffn_cfg
        self.norm_cfg = norm_cfg
        self._init_layers()    

    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cls_cross_attn = MultiheadAttention(**self.cls_cross_attn_cfg)
        self.box_cross_attn = MultiScaleDeformableAttention(**self.box_cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.cls_ffn = FFN(**self.ffn_cfg)
        self.box_ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(5)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                query_pos: Tensor,
                cls_key: Tensor,
                
                box_value: Tensor,
                reference_points: Tensor,
                spatial_shapes: Tensor,
                level_start_index: Tensor,
                valid_ratios: Tensor,
                reg_branches: Optional[nn.Module] = None,
                
                cls_key_pos: Tensor = None,
                cls_cross_attn_mask: Tensor = None,
                cls_key_padding_mask: Tensor = None,
                
                self_attn_mask: Tensor = None,

                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
    
        query = self.self_attn(
            query=query,
            key=query,
            value=query,  
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        box_query = self.box_cross_attn(
            query=query,
            query_pos=query_pos,
            value=box_value, 
            reference_points = reference_points,
            spatial_shapes = spatial_shapes,
            level_start_index = level_start_index,
            valid_ratios = valid_ratios,
            reg_branches = reg_branches,
            **kwargs)
        box_query = self.norms[1](box_query)
        box_query = self.box_ffn(box_query)
        box_query = self.norms[2](box_query)

        #with torch.autocast('cuda'):
        cls_query = self.cls_cross_attn(
            query=query,
            key=cls_key,
            value=cls_key, #+ key_pos   
            query_pos=query_pos,
            key_pos=cls_key_pos,
            attn_mask=cls_cross_attn_mask,
            key_padding_mask=cls_key_padding_mask,
            **kwargs)
        cls_query = self.norms[3](cls_query)
        cls_query = self.cls_ffn(cls_query)
        cls_query = self.norms[4](cls_query)
        
        return box_query, cls_query


@MODELS.register_module()
class CenterTransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                 #pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 **kwargs):
        super().__init__(**kwargs)
        
        #self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        #self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        
    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = PETRMultiheadFlashAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(3)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                key: Tensor = None,
                value: Tensor = None,
                query_pos: Tensor = None,
                key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                cross_attn_mask: Tensor = None,
                key_padding_mask: Tensor = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
        query = self.self_attn(
            query=query,
            key=query,
            value=query,    # + query_pos
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        with torch.autocast('cuda'):
            query = self.cross_attn(
                query=query,
                key=key,
                value=key, #+ key_pos
                query_pos=query_pos,
                key_pos=key_pos,
                attn_mask=cross_attn_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        query = self.norms[1](query)
        query = self.ffn(query)
        query = self.norms[2](query)

        #query = query.transpose(1, 2)
        return query




@MODELS.register_module()
class MFTransformerDecoderLayer(DetrTransformerDecoderLayer):

    def __init__(self,
                 #pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
                 **kwargs):
        super().__init__(**kwargs)
        
        #self.self_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        #self.cross_posembed = PositionEncodingLearned(**pos_encoding_cfg)
        
    def _init_layers(self) -> None:
        """Initialize self-attention, FFN, and normalization."""
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cls_cross_attn = PETRMultiheadFlashAttention(**self.cross_attn_cfg)
        self.box_cross_attn = PETRMultiheadFlashAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.cls_ffn = FFN(**self.ffn_cfg)
        self.box_ffn = FFN(**self.ffn_cfg)
        norms_list = [
            build_norm_layer(self.norm_cfg, self.embed_dims)[1]
            for _ in range(5)
        ]
        self.norms = ModuleList(norms_list)

    def forward(self,
                query: Tensor,
                query_pos: Tensor = None,
                box_key: Tensor = None,
                cls_key: Tensor = None,
               
                box_key_pos: Tensor = None,
                cls_key_pos: Tensor = None,
                self_attn_mask: Tensor = None,
                box_cross_attn_mask: Tensor = None,
                cls_cross_attn_mask: Tensor = None,
                box_key_padding_mask: Tensor = None,
                cls_key_padding_mask: Tensor = None,
                #lidar_tokens_num: int = None,
                **kwargs) -> Tensor:
        """
        Args:
            query (Tensor): The input query, has shape (bs, num_queries, dim).
            key (Tensor, optional): The input key, has shape (bs, num_keys,
                dim). If `None`, the `query` will be used. Defaults to `None`.
            value (Tensor, optional): The input value, has the same shape as
                `key`, as in `nn.MultiheadAttention.forward`. If `None`, the
                `key` will be used. Defaults to `None`.
            query_pos (Tensor, optional): The positional encoding for `query`,
                has the same shape as `query`. If not `None`, it will be added
                to `query` before forward function. Defaults to `None`.
            key_pos (Tensor, optional): The positional encoding for `key`, has
                the same shape as `key`. If not `None`, it will be added to
                `key` before forward function. If None, and `query_pos` has the
                same shape as `key`, then `query_pos` will be used for
                `key_pos`. Defaults to None.
            self_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            cross_attn_mask (Tensor, optional): ByteTensor mask, has shape
                (num_queries, num_keys), as in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor, optional): The `key_padding_mask` of
                `self_attn` input. ByteTensor, has shape (bs, num_value).
                Defaults to None.

        Returns:
            Tensor: forwarded results, has shape (bs, num_queries, dim).
        """
    
        query = self.self_attn(
            query=query,
            key=query,
            value=query,    # + query_pos
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask,
            **kwargs)
        query = self.norms[0](query)
        # Note that the `value` (equal to `key`) is encoded with `key_pos`.
        # This is different from the standard DETR Decoder Layer.
        # lidar_keys, image_keys = key[:, :lidar_tokens_num], key[:, lidar_tokens_num:]
        # lidar_keys_pos, image_keys_pos = key_pos[:, :lidar_tokens_num], key_pos[:, lidar_tokens_num:]
        # lidar_keys_padding_mask, image_keys_padding_mask = key_padding_mask[:, :lidar_tokens_num], key_padding_mask[:, lidar_tokens_num:]
        with torch.autocast('cuda'):
            box_query = self.box_cross_attn(
                query=query,
                key=box_key,
                value=box_key, #+ key_pos
                query_pos=query_pos,
                key_pos=box_key_pos,
                attn_mask=box_cross_attn_mask,
                key_padding_mask=box_key_padding_mask,
                **kwargs)
        box_query = self.norms[1](box_query)
        box_query = self.box_ffn(box_query)
        box_query = self.norms[2](box_query)

        with torch.autocast('cuda'):
            cls_query = self.cls_cross_attn(
                query=query,
                key=cls_key,
                value=cls_key, #+ key_pos
                query_pos=query_pos,
                key_pos=cls_key_pos,
                attn_mask=cls_cross_attn_mask,
                key_padding_mask=cls_key_padding_mask,
                **kwargs)
        cls_query = self.norms[3](cls_query)
        cls_query = self.cls_ffn(cls_query)
        cls_query = self.norms[4](cls_query)
        
        #query = query.transpose(1, 2)
        return box_query, cls_query
