from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

from mmengine import Config, load
from mmengine.logging import MMLogger

from mmdet3d.registry import METRICS
from mmdet3d.evaluation import NuScenesMetric



@METRICS.register_module()
class MFNuScenesMetric(NuScenesMetric):
    
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 **kwargs
                 ) -> None:
        
        super().__init__(data_root, ann_file, **kwargs)
        
    
    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            pred_3d = data_sample['pred_instances_3d']
            pred_2d = data_sample['pred_instances']
            for attr_name in pred_3d:
                pred_3d[attr_name] = pred_3d[attr_name].to('cpu')
            result['pred_instances_3d'] = pred_3d
            for attr_name in pred_2d:
                pred_2d[attr_name] = pred_2d[attr_name].to('cpu')
            result['pred_instances'] = pred_2d
            sample_idx = data_sample['sample_idx']
            result['sample_idx'] = sample_idx
            result['additional_metric'] = data_sample['additional_metric']
            self.results.append(result)


    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:

        metric_dict = super().compute_metrics( results )
        
        
        # classes = self.dataset_meta['classes']
        # result_dict, tmp_dir = self.format_results(results, classes,
        #                                            self.jsonfile_prefix)
        

        custom_metric_dict = self.compute_custom_metrics( results )
        
        metric_dict.update( custom_metric_dict )
        # logger: MMLogger = MMLogger.get_current_instance()

        # classes = self.dataset_meta['classes']
        # self.version = self.dataset_meta['version']
        # # load annotations
        # self.data_infos = load(
        #     self.ann_file, backend_args=self.backend_args)['data_list']
        # result_dict, tmp_dir = self.format_results(results, classes,
        #                                            self.jsonfile_prefix)

        # metric_dict = {}
        # print(result_dict.keys())
        # print(result_dict['pred_instances_3d'])

        # if self.format_only:
        #     logger.info(
        #         f'results are saved in {osp.basename(self.jsonfile_prefix)}')
        #     return metric_dict

        # for metric in self.metrics:
        #     ap_dict = self.nus_evaluate(
        #         result_dict, classes=classes, metric=metric, logger=logger)
        #     for result in ap_dict:
        #         metric_dict[result] = ap_dict[result]

        # if tmp_dir is not None:
        #     tmp_dir.cleanup()
        return metric_dict
    
    def compute_custom_metrics( self, results: dict ):
        
        # print( '##################' )
        # print(len(results))
        # print( results[0].keys() )
        # print( '##################' )
        # l1_dist, l1_dist_x, l1_dist_y, l1_dist_z = 0.0, 0.0, 0.0, 0.0
        # front_recall, detect_recall = 0.0, 0.0
        front_recall = 0.0
        for result in results:
            custom_metric = result['additional_metric']
            #print( custom_metric )
            # l1_dist += custom_metric['l1_dist']
            # l1_dist_x+= custom_metric['l1_dist_x']
            # l1_dist_y+= custom_metric['l1_dist_y']
            # l1_dist_z+= custom_metric['l1_dist_z']
            # detect_recall +=  custom_metric['detect_recall']
            front_recall += custom_metric['front_recall']
            
            
        
        # l1_dist /= len(results)
        # l1_dist_x /= len(results)
        # l1_dist_y /= len(results)
        # l1_dist_z /= len(results)
        # detect_recall /= len(results)
        front_recall /= len(results)
        
               
        
        return dict(
            front_recall = front_recall,
            # l1_dist = l1_dist,
            # l1_dist_x = l1_dist_x,
            # l1_dist_y = l1_dist_y,
            # l1_dist_z = l1_dist_z,
            # detect_recall = detect_recall,
        )