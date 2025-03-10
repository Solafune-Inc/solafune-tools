from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from rasterio.features import rasterize
from typing import Union

def bbox_to_polygon(bbox: Union[tuple, list]) -> Polygon:
    """
    Converts a bounding box into a polygon.

    Parameters:
        bbox (list): [x_min, y_min, width, height]

    Returns:
        Polygon: Polygon object representing the bounding box.
    """
    x_min, y_min, width, height = bbox
    return Polygon([
        (x_min, y_min), (x_min + width, y_min), (x_min + width, y_min + height), (x_min, y_min + height)
    ])

class IOUBasedMetrics:
    def __init__(self) -> None:
        """
        Initializes the IOUBasedMetrics class.
        """
        pass

    def getIOU(self, polygon1: Polygon, polygon2: Polygon) -> float:
        """
        Computes the Intersection over Union (IoU) between two polygons.
        """
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        return intersection / union if union > 0 else 0

    def match_polygons(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Matches ground truth polygons with predicted polygons based on IoU threshold.
        """
        matched_instances = {}
        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))
        
        for gt_idx, gt_polygon in enumerate(gt_polygons):
            best_iou = iou_threshold
            best_pred_idx = None
            
            for pred_idx, pred_polygon in enumerate(pred_polygons):
                iou = self.getIOU(gt_polygon, pred_polygon)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            
            if best_pred_idx is not None:
                matched_instances[(gt_idx, best_pred_idx)] = best_iou
                gt_matched[gt_idx] = 1
                pred_matched[best_pred_idx] = 1
        
        return matched_instances, gt_matched, pred_matched

    def compute_pq(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).
        """
        matched_instances, _, _ = self.match_polygons(gt_polygons, pred_polygons, iou_threshold)
        
        sq_sum = sum(matched_instances.values())
        num_matches = len(matched_instances)
        sq = sq_sum / num_matches if num_matches else 0
        rq = num_matches / ((len(gt_polygons) + len(pred_polygons)) / 2.0) if (gt_polygons or pred_polygons) else 0
        pq = sq * rq
        
        return pq, sq, rq
    
    def compute_f1(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Compute the F1 score, precision, and recall for the given ground truth and predicted polygons.
        """
        matched_instances, gt_matched, pred_matched = self.match_polygons(gt_polygons, pred_polygons, iou_threshold)
        
        tp = len(matched_instances)
        fp = len(pred_polygons) - tp
        fn = len(gt_polygons) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # if no prediction, precision is considered as 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # if no ground truth, recall is considered as 1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # if either precision or recall is 0, f1 is 0
        
        return f1, precision, recall
    
    def compute_map(self, gt_polygons: list, pred_polygons: list, iou_thresholds=None) -> float:
        """
        Compute mean Average Precision (mAP) over a range of IoU thresholds.
        If no thresholds are provided, it defaults to [0.5, 1.0, 0.05].
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.95, 0.05]
        elif isinstance(iou_thresholds, (float, int)):  # Handle single threshold case
            iou_thresholds = [iou_thresholds]

        average_precisions = []
        
        for iou_threshold in iou_thresholds:
            matched_instances, _, _ = self.match_polygons(gt_polygons, pred_polygons, iou_threshold)
            tp = len(matched_instances)
            fp = len(pred_polygons) - tp
            fn = len(gt_polygons) - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precisions = [precision]
            recalls = [recall]
            
            for threshold in np.linspace(0, 1, 11):  # Standard 11-point interpolation
                precisions.append(max([p for r, p in zip(recalls, precisions) if r >= threshold], default=0))
            
            ap = np.mean(precisions)
            average_precisions.append(ap)

        return np.mean(average_precisions) if average_precisions else 0.0
    
class PixelBasedMetrics:
    def __init__(self) -> None:
        pass

    def polygons_to_mask(self, polygons, array_dim) -> np.ndarray:
        """
        Converts a list of polygons into a binary mask.
        
        Args:
            polygons (list): List of polygons, where each polygon is represented by a list of (x, y) tuples.
            array_dim (tuple): Dimensions of the output mask (height, width).
        
        Returns:
            np.ndarray: Binary mask with 1s for polygon areas and 0s elsewhere.
        """
        shapes = [(polygon, 1) for polygon in polygons]
        mask = rasterize(shapes, out_shape=array_dim, fill=0, dtype=np.uint8)
        return mask

    def compute_f1(self, gt_polygons, pred_polygons, array_dim=(1024, 1024)) -> tuple:
        """
        Compute the F1 score, precision, and recall for the given ground truth and predicted polygons.
        
        Args:
            gt_polygons (list): List of ground truth polygons.
            pred_polygons (list): List of predicted polygons.
            array_dim (tuple, optional): Dimensions of the output mask (height, width). Defaults to (1024, 1024).

        Returns:
            tuple: A tuple containing the F1 score, precision, and recall.
        """
        # Pixel-level improvement
        # Create binary masks for ground truth and predictions
        gt_mask = self.polygons_to_mask(gt_polygons, array_dim)
        pred_mask = self.polygons_to_mask(pred_polygons, array_dim)
        
        # Calculate pixel-level True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = np.sum((gt_mask == 1) & (pred_mask == 1))
        fp = np.sum((gt_mask == 0) & (pred_mask == 1))
        fn = np.sum((gt_mask == 1) & (pred_mask == 0))
        
        # Calculate precision, recall, and F1 score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # if no prediction, precision is considered as 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # if no ground truth, recall is considered as 1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # if either precision or recall is 0, f1 is 0
        
        return f1, precision, recall