# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Optional

import mmcv
import numpy as np
from mmengine.fileio import get

from mmdet3d.datasets.transforms import LoadMultiViewImageFromFiles
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class BEVLoadMultiViewImageFromFiles(LoadMultiViewImageFromFiles):
    """Load multi channel images from a list of separate channel files.

    ``BEVLoadMultiViewImageFromFiles`` adds the following keys for the
    convenience of view transforms in the forward:
        - 'cam2lidar'
        - 'lidar2img'

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
        num_views (int): Number of view in a frame. Defaults to 5.
        num_ref_frames (int): Number of frame in loading. Defaults to -1.
        test_mode (bool): Whether is test mode in loading. Defaults to False.
        set_default_scale (bool): Whether to set default scale.
            Defaults to True.
    """

    def transform(self, results: dict) -> Optional[dict]:
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
            Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        # TODO: consider split the multi-sweep part out of this pipeline
        # Derive the mask and transform for loading of multi-sweep data
        if self.num_ref_frames > 0:
            # init choice with the current frame
            init_choice = np.array([0], dtype=np.int64)
            num_frames = len(results['img_filename']) // self.num_views - 1
            if num_frames == 0:  # no previous frame, then copy cur frames
                choices = np.random.choice(
                    1, self.num_ref_frames, replace=True)
            elif num_frames >= self.num_ref_frames:
                # NOTE: suppose the info is saved following the order
                # from latest to earlier frames
                if self.test_mode:
                    choices = np.arange(num_frames - self.num_ref_frames,
                                        num_frames) + 1
                # NOTE: +1 is for selecting previous frames
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=False) + 1
            elif num_frames > 0 and num_frames < self.num_ref_frames:
                if self.test_mode:
                    base_choices = np.arange(num_frames) + 1
                    random_choices = np.random.choice(
                        num_frames,
                        self.num_ref_frames - num_frames,
                        replace=True) + 1
                    choices = np.concatenate([base_choices, random_choices])
                else:
                    choices = np.random.choice(
                        num_frames, self.num_ref_frames, replace=True) + 1
            else:
                raise NotImplementedError
            choices = np.concatenate([init_choice, choices])
            select_filename = []
            for choice in choices:
                select_filename += results['img_filename'][choice *
                                                           self.num_views:
                                                           (choice + 1) *
                                                           self.num_views]
            results['img_filename'] = select_filename
            for key in ['cam2img', 'lidar2cam']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += results[key][choice *
                                                       self.num_views:(choice +
                                                                       1) *
                                                       self.num_views]
                    results[key] = select_results
            for key in ['ego2global']:
                if key in results:
                    select_results = []
                    for choice in choices:
                        select_results += [results[key][choice]]
                    results[key] = select_results
            # Transform lidar2cam to
            # [cur_lidar]2[prev_img] and [cur_lidar]2[prev_cam]
            for key in ['lidar2cam']:
                if key in results:
                    # only change matrices of previous frames
                    for choice_idx in range(1, len(choices)):
                        pad_prev_ego2global = np.eye(4)
                        prev_ego2global = results['ego2global'][choice_idx]
                        pad_prev_ego2global[:prev_ego2global.
                                            shape[0], :prev_ego2global.
                                            shape[1]] = prev_ego2global
                        pad_cur_ego2global = np.eye(4)
                        cur_ego2global = results['ego2global'][0]
                        pad_cur_ego2global[:cur_ego2global.
                                           shape[0], :cur_ego2global.
                                           shape[1]] = cur_ego2global
                        cur2prev = np.linalg.inv(pad_prev_ego2global).dot(
                            pad_cur_ego2global)
                        for result_idx in range(choice_idx * self.num_views,
                                                (choice_idx + 1) *
                                                self.num_views):
                            results[key][result_idx] = \
                                results[key][result_idx].dot(cur2prev)
        # Support multi-view images with different shapes
        # TODO: record the origin shape and padded shape
        filename, cam2img, lidar2cam, cam2lidar, lidar2img = [], [], [], [], []
        for _, cam_item in results['images'].items():
            filename.append(cam_item['img_path'])
            lidar2cam.append(cam_item['lidar2cam'])

            lidar2cam_array = np.array(cam_item['lidar2cam']).astype(
                np.float32)
            lidar2cam_rot = lidar2cam_array[:3, :3]
            lidar2cam_trans = lidar2cam_array[:3, 3:4]
            camera2lidar = np.eye(4)
            camera2lidar[:3, :3] = lidar2cam_rot.T
            camera2lidar[:3, 3:4] = -1 * np.matmul(
                lidar2cam_rot.T, lidar2cam_trans.reshape(3, 1))
            cam2lidar.append(camera2lidar)

            cam2img_array = np.eye(4).astype(np.float32)
            cam2img_array[:3, :3] = np.array(cam_item['cam2img']).astype(
                np.float32)
            cam2img.append(cam2img_array)
            lidar2img.append(cam2img_array @ lidar2cam_array)

        results['img_path'] = filename
        results['cam2img'] = np.stack(cam2img, axis=0)
        results['lidar2cam'] = np.stack(lidar2cam, axis=0)
        results['cam2lidar'] = np.stack(cam2lidar, axis=0)
        results['lidar2img'] = np.stack(lidar2img, axis=0)

        results['ori_cam2img'] = copy.deepcopy(results['cam2img'])

        # img is of shape (h, w, c, num_views)
        # h and w can be different for different views
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        imgs = [
            mmcv.imfrombytes(
                img_byte,
                flag=self.color_type,
                backend='pillow',
                channel_order='rgb') for img_byte in img_bytes
        ]
        # handle the image with different shape
        img_shapes = np.stack([img.shape for img in imgs], axis=0)
        img_shape_max = np.max(img_shapes, axis=0)
        img_shape_min = np.min(img_shapes, axis=0)
        assert img_shape_min[-1] == img_shape_max[-1]
        if not np.all(img_shape_max == img_shape_min):
            pad_shape = img_shape_max[:2]
        else:
            pad_shape = None
        if pad_shape is not None:
            imgs = [
                mmcv.impad(img, shape=pad_shape, pad_val=0) for img in imgs
            ]
        img = np.stack(imgs, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape[:2]
        if self.set_default_scale:
            results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        results['num_views'] = self.num_views
        results['num_ref_frames'] = self.num_ref_frames
        return results

@TRANSFORMS.register_module()
class MFMultiViewImageFeatureCorrespondingPoints:
    
    def __init__(self, input_size, downsample=1, min_dist=1e-5, max_dist=None):
        self.input_size = input_size
        self.downsample = downsample
        self.min_dist = min_dist
        self.max_dist = max_dist
    
    def mask_points_by_range(self, points_3d, points_2d, depths, img_size):
        """
        Args:
            points2d: (N, 2)
            depths:   (N, )
            img_size: (H, W)
        Returns:
            points2d: (N', 2)
            depths:   (N', )
        """
        H, W = img_size
        mask = np.ones(depths.shape, dtype=np.bool_)
        mask = np.logical_and(mask, points_2d[:, 0] >= 0)
        mask = np.logical_and(mask, points_2d[:, 0] < W)
        mask = np.logical_and(mask, points_2d[:, 1] >= 0)
        mask = np.logical_and(mask, points_2d[:, 1] < H)
        points_2d = points_2d[mask]
        depths = depths[mask]
        points_3d = points_3d[mask]
        return points_3d, points_2d, depths
    
    def mask_points_by_dist(self, points_3d, points_2d, depths, min_dist, max_dist):
        mask = np.ones(depths.shape, dtype=np.bool_)
        mask = np.logical_and(mask, depths >= min_dist)
        if max_dist is not None:
            mask = np.logical_and(mask, depths <= max_dist)
        points_2d = points_2d[mask]
        depths = depths[mask]
        points_3d = points_3d[mask]
        return points_3d, points_2d, depths
    
    def __call__(self, data: dict ) -> dict :
        
        #assert 'points' in data
        points = data['points'].tensor[:, :3].numpy() 
        H, W = self.input_size
        dH, dW = H // self.downsample, W // self.downsample
        # img_shape = data['img_shape']
        # print('img_shape:', img_shape )
        # pad_shape = data['pad_shape']
        # print('pad_shape:', pad_shape )
        #lidar2imgs = data['lidar2img']
        points_lidar = np.concatenate([points,  np.ones((points.shape[0], 1))], axis=-1)
        points_maps, points_map_masks = [], []
        for i, lidar2img in enumerate( data['lidar2img']):
            points_image = points_lidar.dot(lidar2img.T)
            points_image = points_image[:, :3]  # (N_points, 3)     3: (du, dv, d)
            points_2d = points_image[:, :2] / points_image[:, 2:3]  # (N_points, 2)     2: (u, v)
            points_2d = np.round(points_2d / self.downsample)
            depths = points_image[:, 2]  # (N_points, )
            
            valid_points = np.copy(points)
            valid_points, points_2d, depths = self.mask_points_by_range(valid_points, points_2d, depths, (dH, dW) )
            valid_points, points_2d, depths = self.mask_points_by_dist(valid_points, points_2d, depths, self.min_dist, self.max_dist)
            
            ranks = points_2d[:, 0] + points_2d[:, 1] * dW
            sort = (ranks + depths / 1000.).argsort()
            points_2d, depths, ranks = points_2d[sort], depths[sort], ranks[sort]
            valid_points = valid_points[sort]

            kept = np.ones(points_2d.shape[0], dtype=np.bool_)
            kept[1:] = (ranks[1:] != ranks[:-1])
            points_2d, depths = points_2d[kept], depths[kept]
            points_2d = points_2d.astype(np.int_)
            valid_points = valid_points[kept]
            
            points_map = np.zeros((dH, dW, 3), dtype=np.float32)
            points_map[points_2d[:, 1], points_2d[:, 0], :] = valid_points
            #print('points_map:', points_map.shape )
            points_map_mask = np.zeros((dH, dW), dtype=np.bool_)
            points_map_mask[points_2d[:, 1], points_2d[:, 0]] = 1
            points_maps.append(points_map)
            points_map_masks.append(points_map_mask)
            
        points_map = np.stack(points_maps, axis=0)
        points_map_mask = np.stack(points_map_masks, axis=0)
        
        data['points_map'] = points_map
        data['points_map_mask'] = points_map_mask
        
        return data