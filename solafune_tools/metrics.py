from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from rasterio.features import rasterize

class IOUBasedMetrics:
    def __init__(self) -> None:
        """
        Initializes the IOUBasedMetrics class.
        
        This class provides methods to compute metrics based on the Intersection over Union (IoU) 
        between polygons, such as precision, recall, F1 score, and Panoptic Quality (PQ).
        """
        pass

    def getIOU(self, polygon1: Polygon, polygon2: Polygon) -> float:
        """
        Computes the Intersection over Union (IoU) between two polygons.
        Parameters
        ----------
        polygon1 : Polygon
            The first polygon.
        polygon2 : Polygon
            The second polygon.
        Returns
        -------
        float
            The IoU value between the two polygons.
        """
        intersection = polygon1.intersection(polygon2).area
        union = polygon1.union(polygon2).area
        if union == 0:
            return 0
        return intersection / union

    def compute_pq(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Compute the Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) 
        for the given ground truth and predicted polygons.

        Args:
            gt_polygons (list): List of ground truth polygons.
            pred_polygons (list): List of predicted polygons.
            matched_instances (dict): Dictionary of matched instances with IoU values.

        Returns:
            tuple: A tuple containing the PQ, SQ, and RQ.
        """
        matched_instances = {}
        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))

        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))
        for gt_idx, gt_polygon in tqdm(enumerate(gt_polygons)):
            best_iou = iou_threshold
            best_pred_idx = None
            for pred_idx, pred_polygon in enumerate(pred_polygons):
                # if gt_matched[gt_idx] == 1 or pred_matched[pred_idx] == 1:
                #     continue
                
                iou = self.getIOU(gt_polygon, pred_polygon)
                if iou == 0:
                    continue
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            if best_pred_idx is not None:
                matched_instances[(gt_idx, best_pred_idx)] = best_iou
                gt_matched[gt_idx] = 1
                pred_matched[best_pred_idx] = 1

        
        sq_sum = sum(matched_instances.values())
        num_matches = len(matched_instances)
        sq = sq_sum / num_matches if num_matches else 0
        rq = num_matches / ((len(gt_polygons) + len(pred_polygons))/2.0) if (gt_polygons or pred_polygons) else 0
        pq = sq * rq

        return pq, sq, rq
    
    def compute_f1(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Compute the F1 score, precision, and recall for the given ground truth and predicted polygons.
    
        Args:
            gt_polygons (list): List of ground truth polygons.
            pred_polygons (list): List of predicted polygons.
            iou_threshold (float, optional): Intersection over Union (IoU) threshold to consider a match. Defaults to 0.5.
    
        Returns:
            tuple: A tuple containing the F1 score, precision, and recall.
        """
        matched_instances = {}
        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))

        # IoU計算とマッチング候補の特定
        gt_matched = np.zeros(len(gt_polygons))
        pred_matched = np.zeros(len(pred_polygons))
        for gt_idx, gt_polygon in enumerate(gt_polygons):
            best_iou = iou_threshold
            best_pred_idx = None
            for pred_idx, pred_polygon in enumerate(pred_polygons):
                # if gt_matched[gt_idx] == 1 or pred_matched[pred_idx] == 1:
                #     continue
                
                iou = self.getIOU(gt_polygon, pred_polygon)
                if iou == 0:
                    continue
                
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = pred_idx
            if best_pred_idx is not None:
                matched_instances[(gt_idx, best_pred_idx)] = best_iou
                gt_matched[gt_idx] = 1
                pred_matched[best_pred_idx] = 1

        # F1, Precision, Recall
        
        tp = len(matched_instances)
        fp = len(pred_polygons) - tp
        fn = len(gt_polygons) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # if no prediction, precision is considered as 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # if no ground truth, recall is considered as 1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0  # if either precision or recall is 0, f1 is 0

        return f1, precision, recall
    
class PixelBasedMetrics:
    def __init__(self) -> None:
        pass

    def polygons_to_mask(self, polygons, array_dim):
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

    def compute_f1(self, gt_polygons, pred_polygons, array_dim=(1024, 1024)):
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