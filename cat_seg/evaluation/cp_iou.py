import os
import numpy as np
import pandas as pd

import itertools
import json
import numpy as np
import os
from collections import OrderedDict
import PIL.Image as Image
import torch
import pandas as pd

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
import copy


# self.results = {thresh: {'TP': 0.0, 'FN': 0.0, 'FP': 0.0} for thresh in [0.1 * i for i in range(1, 10)]}
# self.counts = {thresh: {'TP': 0.0, 'FN': 0.0, 'FP': 0.0} for thresh in [0.1 * i for i in range(1, 10)]}        
# self.cot = 0


def compute_iou(self, mask1, mask2):
    assert mask1.shape == mask2.shape, "The shape of both masks must be the same."
    mask1 = mask1 > 0
    mask2 = mask2 > 0
    
    # aviod cuda OOM
    if mask1.device != mask2.device:
        mask1 = mask1.detach().cpu()
        mask2 = mask2.detach().cpu()  
              
    intersection = (mask1 & mask2).sum().float()
    union = (mask1 | mask2).sum().float()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou.item()

def CP_mIoU_all_thres(self, pred, targets):
    pred = F.interpolate(pred.unsqueeze(0), size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
    num_classes = pred.shape[1]
    mask = targets != self.sem_seg_head.ignore_value
    pred = pred.permute(0,2,3,1)
    _targets = torch.zeros(pred.shape, device=self.device)
    _onehot = F.one_hot(targets[mask], num_classes=num_classes).float().to(self.device)
    _targets[0, mask] = _onehot
    # Initialize a dictionary to store results for each threshold
    results = {thresh: {'TP': 0.0, 'FN': 0.0, 'FP': 0.0} for thresh in [0.1 * i for i in range(1, 10)]}
    counts = {thresh: {'TP': 0.0, 'FN': 0.0, 'FP': 0.0} for thresh in [0.1 * i for i in range(1, 10)]}
    _targets = _targets[0].permute(2,0,1)
    w, h = pred.shape[1], pred.shape[2]
    for thresh in results.keys():
        # Apply threshold
        pred_thresh = torch.where(pred[0,:].permute(2, 0, 1) > thresh, 1.0, 0.0)
        
        for i in range(pred_thresh.shape[0]):
            if 1 not in pred_thresh[i,:] and 1 not in _targets[i,:]:
                continue                
            if 1 in pred_thresh[i,:] and 1 in _targets[i,:]:
                results[thresh]['TP'] += self.compute_iou(pred_thresh[i,:], _targets[i,:])
                counts[thresh]['TP'] += 1
            elif 1 in pred_thresh[i,:] and 1 not in _targets[i,:]:
                results[thresh]['FP'] += torch.sum(pred_thresh[i,:].detach().cpu() == 1) / (w * h)
                counts[thresh]['FP'] += 1
            elif 1 not in pred_thresh[i,:] and 1 in _targets[i,:]:
                results[thresh]['FN'] += torch.sum(_targets[i,:].detach().cpu() == 1) / (w * h)
                counts[thresh]['FN'] += 1
    # Average the results
    for thresh in results.keys():
        if counts[thresh]['TP'] != 0:
            results[thresh]['TP'] /= counts[thresh]['TP']
        if counts[thresh]['FN'] != 0:
            results[thresh]['FN'] /= counts[thresh]['FN']
        if counts[thresh]['FP'] != 0:
            results[thresh]['FP'] /= counts[thresh]['FP']
    # with open("/home/qiming/Desktop/code/iclr25/demo/CAT-Seg/output/res/res.txt", "a") as fr:
    #     for thresh in results.keys():
    #         fr.write(f"Threshold: {thresh}, TP: {results[thresh]['TP']}, FN: {results[thresh]['FN']}, FP: {results[thresh]['FP']}\n")
    # Accumulate results
    for thresh in results.keys():
        self.results[thresh]['TP'] += results[thresh]['TP']
        self.results[thresh]['FN'] += results[thresh]['FN']
        self.results[thresh]['FP'] += results[thresh]['FP']
    self.cot += 1
    if self.cot >= 5103:
        print(f"here we come : {self.cot}")
        for thresh in self.results.keys():
            self.results[thresh]['TP'] /= 5104
            self.results[thresh]['FN'] /= 5104
            self.results[thresh]['FP'] /= 5104  
        # Convert self.results to JSON-serializable format
        results_json = {thresh: {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in metrics.items()} 
                        for thresh, metrics in self.results.items()}
        # Save self.results to a JSON file
        with open("./res.json", "w") as json_file:
            json.dump(results_json, json_file, indent=4)        
  
    return results

import os
import json
import torch
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from detectron2.evaluation import SemSegEvaluator

class CPmIoUEvaluator(SemSegEvaluator):
    def __init__(
        self, dataset_name, distributed=True, output_dir=None, num_classes=None, ignore_label=None
    ):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )
        self.results = {thresh: {'TP': 0.0, 'FN': 0.0, 'FP': 0.0} for thresh in [0.1 * i for i in range(1, 10)]}
        self.counts = {thresh: {'TP': 0.0, 'FN': 0.0, 'FP': 0.0} for thresh in [0.1 * i for i in range(1, 10)]}
        self.cot = 0
    
    def compute_iou(self, mask1, mask2):
        assert mask1.shape == mask2.shape, "The shape of both masks must be the same."
        mask1 = mask1 > 0
        mask2 = mask2 > 0
        
        if mask1.device != mask2.device:
            mask1 = mask1.cpu()
            mask2 = mask2.cpu()
        
        intersection = (mask1 & mask2).sum().float()
        union = (mask1 | mask2).sum().float()
        return (intersection / union).item() if union != 0 else 0.0
    
    def process(self, pred, targets):
        pred = F.interpolate(pred.unsqueeze(0), size=(targets.shape[-2], targets.shape[-1]), mode="bilinear", align_corners=False)
        num_classes = pred.shape[1]
        mask = targets != self._ignore_label
        pred = pred.permute(0, 2, 3, 1)
        
        _targets = torch.zeros(pred.shape, device=pred.device)
        _onehot = F.one_hot(targets[mask], num_classes=num_classes).float().to(pred.device)
        _targets[0, mask] = _onehot
        _targets = _targets[0].permute(2, 0, 1)
        w, h = pred.shape[1], pred.shape[2]
        
        for thresh in self.results.keys():
            pred_thresh = torch.where(pred[0, :].permute(2, 0, 1) > thresh, 1.0, 0.0)
            for i in range(pred_thresh.shape[0]):
                if 1 not in pred_thresh[i, :] and 1 not in _targets[i, :]:
                    continue
                if 1 in pred_thresh[i, :] and 1 in _targets[i, :]:
                    self.results[thresh]['TP'] += self.compute_iou(pred_thresh[i, :], _targets[i, :])
                    self.counts[thresh]['TP'] += 1
                elif 1 in pred_thresh[i, :] and 1 not in _targets[i, :]:
                    self.results[thresh]['FP'] += torch.sum(pred_thresh[i, :].cpu() == 1) / (w * h)
                    self.counts[thresh]['FP'] += 1
                elif 1 not in pred_thresh[i, :] and 1 in _targets[i, :]:
                    self.results[thresh]['FN'] += torch.sum(_targets[i, :].cpu() == 1) / (w * h)
                    self.counts[thresh]['FN'] += 1
        
        self.cot += 1
        if self.cot >= 5103:
            for thresh in self.results.keys():
                self.results[thresh]['TP'] /= 5104
                self.results[thresh]['FN'] /= 5104
                self.results[thresh]['FP'] /= 5104
            
            results_json = {thresh: {k: v if isinstance(v, float) else v.item() for k, v in metrics.items()} 
                            for thresh, metrics in self.results.items()}
            
            if self._output_dir:
                os.makedirs(self._output_dir, exist_ok=True)
                with open(os.path.join(self._output_dir, "all_thres_res.json"), "w") as json_file:
                    json.dump(results_json, json_file, indent=4)
        
        return self.results
    
    def evaluate(self):
        return OrderedDict({"CPmIoU": self.results})
