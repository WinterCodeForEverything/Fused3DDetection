from typing import List, Optional, Tuple, Union

import torch
from mmcv.cnn import ConvModule
from mmcv.ops import GroupAll
from mmcv.ops import PointsSampler as Points_Sampler
from mmcv.ops import QueryAndGroup, gather_points
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.registry import MODELS
#from .builder import SA_MODULES

@MODELS.register_module()
class MFPointSAModule(nn.Module):
    """Base module for point set abstraction module used in PointNets.

    Args:
        num_point (int): Number of points.
        radii (List[float]): List of radius in each ball query.
        sample_nums (List[int]): Number of samples in each ball query.
        mlp_channels (List[List[int]]): Specify of the pointnet before
            the global pooling for each scale.
        fps_mod (List[str]): Type of FPS method, valid mod
            ['F-FPS', 'D-FPS', 'FS']. Defaults to ['D-FPS'].

            - F-FPS: using feature distances for FPS.
            - D-FPS: using Euclidean distances of points for FPS.
            - FS: using F-FPS and D-FPS simultaneously.
        fps_sample_range_list (List[int]): Range of points to apply FPS.
            Defaults to [-1].
        dilated_group (bool): Whether to use dilated ball query.
            Defaults to False.
        use_xyz (bool): Whether to use xyz. Defaults to True.
        normalize_xyz (bool): Whether to normalize local XYZ with radius.
            Defaults to False.
        grouper_return_grouped_xyz (bool): Whether to return grouped xyz
            in `QueryAndGroup`. Defaults to False.
        grouper_return_grouped_idx (bool): Whether to return grouped idx
            in `QueryAndGroup`. Defaults to False.
    """

    def __init__(self,
                 num_point: int,
                 radii: List[float],
                 sample_nums: List[int],
                 fps_mod: List[str] = ['D-FPS'],
                 fps_sample_range_list: List[int] = [-1],
                 dilated_group: bool = False,
                 use_xyz: bool = True,
                 normalize_xyz: bool = False,
                 grouper_return_grouped_xyz: bool = False,
                 grouper_return_grouped_idx: bool = False) -> None:
        super(MFPointSAModule, self).__init__()

        assert len(radii) == len(sample_nums)
        assert isinstance(fps_mod, list) or isinstance(fps_mod, tuple)
        assert isinstance(fps_sample_range_list, list) or isinstance(
            fps_sample_range_list, tuple)
        assert len(fps_mod) == len(fps_sample_range_list)

        if isinstance(num_point, int):
            self.num_point = [num_point]
        elif isinstance(num_point, list) or isinstance(num_point, tuple):
            self.num_point = num_point
        elif num_point is None:
            self.num_point = None
        else:
            raise NotImplementedError('Error type of num_point!')

        self.groupers = nn.ModuleList()
        self.fps_mod_list = fps_mod
        self.fps_sample_range_list = fps_sample_range_list

        if self.num_point is not None:
            self.points_sampler = Points_Sampler(self.num_point,
                                                 self.fps_mod_list,
                                                 self.fps_sample_range_list)
        else:
            self.points_sampler = None

        for i in range(len(radii)):
            radius = radii[i]
            sample_num = sample_nums[i]
            if num_point is not None:
                if dilated_group and i != 0:
                    min_radius = radii[i - 1]
                else:
                    min_radius = 0
                grouper = QueryAndGroup(
                    radius,
                    sample_num,
                    min_radius=min_radius,
                    use_xyz=use_xyz,
                    normalize_xyz=normalize_xyz,
                    return_grouped_xyz=grouper_return_grouped_xyz,
                    return_grouped_idx=grouper_return_grouped_idx)
            else:
                grouper = GroupAll(use_xyz)
            self.groupers.append(grouper)

    def _sample_points(self, points_xyz: Tensor, features: Tensor,
                       indices: Tensor, target_xyz: Tensor) -> Tuple[Tensor]:
        """Perform point sampling based on inputs.

        If `indices` is specified, directly sample corresponding points.
        Else if `target_xyz` is specified, use is as sampled points.
        Otherwise sample points using `self.points_sampler`.

        Args:
            points_xyz (Tensor): (B, N, 3) xyz coordinates of the features.
            features (Tensor): (B, C, N) Features of each point.
            indices (Tensor): (B, num_point) Index of the features.
            target_xyz (Tensor): (B, M, 3) new_xyz coordinates of the outputs.

        Returns:
            Tuple[Tensor]:

            - new_xyz: (B, num_point, 3) Sampled xyz coordinates of points.
            - indices: (B, num_point) Sampled points' index.
        """
        xyz_flipped = points_xyz.transpose(1, 2).contiguous()
        if indices is not None:
            assert (indices.shape[1] == self.num_point[0])
            new_xyz = gather_points(xyz_flipped, indices).transpose(
                1, 2).contiguous() if self.num_point is not None else None
        elif target_xyz is not None:
            new_xyz = target_xyz.contiguous()
        else:
            if self.num_point is not None:
                indices = self.points_sampler(points_xyz, features)
                new_xyz = gather_points(xyz_flipped,
                                        indices).transpose(1, 2).contiguous()
            else:
                new_xyz = None

        return new_xyz, indices

    def forward(
        self,
        points_xyz: Tensor,
        features: Optional[Tensor] = None,
        indices: Optional[Tensor] = None,
        target_xyz: Optional[Tensor] = None,
    ) -> Tuple[Tensor]:
    
        # sample points, (B, num_point, 3), (B, num_point)
        new_xyz, indices = self._sample_points(points_xyz, features, indices,
                                               target_xyz)
        # return new_xyz
        center_xyz_list = []
        for i in range(len(self.groupers)):
            # grouped_results may contain:
            # - grouped_features: (B, C, num_point, nsample)
            # - grouped_xyz: (B, 3, num_point, nsample)
            # - grouped_idx: (B, num_point, nsample)
            grouped_results = self.groupers[i](points_xyz, new_xyz, features)
            center_xyz = grouped_results[1].mean(dim=-1).transpose(1, 2)
            center_xyz_list.append(center_xyz)
        return  torch.cat(center_xyz_list, dim=1)
