from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm

def getIOU(polygon1: Polygon, polygon2: Polygon) -> float:
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
class PanopticMetric:
    def __init__(self) -> None:
        """
    A class used to compute panoptic quality metrics for polygon-based segmentation.
    Methods
    -------
    compute_pq(gt_polygons: list, pred_polygons: list, iou_threshold=0.5)
        Computes the Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
        between ground truth and predicted polygons.
    """
        pass

    def compute_pq(self, gt_polygons: list, pred_polygons: list, iou_threshold=0.5) -> tuple:
        """
        Computes the Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ)
        between ground truth and predicted polygons.
        Parameters
        ----------
        gt_polygons : list
            A list of ground truth polygons.
        pred_polygons : list
            A list of predicted polygons.
        iou_threshold : float, optional
            The IoU threshold to consider a match (default is 0.5).
        Returns
        -------
        tuple
            A tuple containing PQ (Panoptic Quality), SQ (Segmentation Quality), and RQ (Recognition Quality) values.
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
                
                iou = getIOU(gt_polygon, pred_polygon)
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

class F1_Metrics:
    def __init__(self) -> None:
        """
        A class used to compute F1 metrics for polygon-based segmentation.
        Methods
        -------
        compute_f1(gt_polygons: list, pred_polygons: list, iou_threshold=0.5)
            Computes the F1 score between ground truth and predicted polygons.
        """
        pass
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
                
                iou = getIOU(gt_polygon, pred_polygon)
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
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1  # if no prediction, precision is considered as 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1  # if no ground truth, recall is considered as 1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # if either precision or recall is 0, f1 is 0
        return f1, precision, recall