from shapely.geometry import Polygon
import numpy as np
from tqdm import tqdm
from rasterio.features import rasterize
from typing import Union, List, Tuple, Dict

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
        Initializes the IOUBasedMetrics class. No parameters are needed.
        """
        pass

    def getIOU(self, polygon1: Polygon, polygon2: Polygon) -> float:
        """
        Computes the Intersection over Union (IoU) between two polygons.

        Args:
            polygon1 (Polygon): The first Shapely Polygon object.
            polygon2 (Polygon): The second Shapely Polygon object.

        Returns:
            float: The IoU value, ranging from 0.0 to 1.0. Returns 0.0 if
                   polygons are invalid or union area is zero.
        """
        if not polygon1.is_valid: polygon1 = polygon1.buffer(0)
        if not polygon2.is_valid: polygon2 = polygon2.buffer(0)
        if not polygon1.is_valid or not polygon2.is_valid: return 0.0

        try:
            intersection_area = polygon1.intersection(polygon2).area
            union_area = polygon1.area + polygon2.area - intersection_area
        except Exception:
            return 0.0 # Handle potential Shapely errors

        # IoU calculation
        return intersection_area / union_area if union_area > 0 else 0.0

    def match_polygons(self, gt_polygons: List[Polygon], pred_polygons: List[Polygon], iou_threshold=0.5) -> Tuple[Dict[Tuple[int, int], float], np.ndarray, np.ndarray]:
        """
        Matches ground truth polygons with predicted polygons based on IoU threshold.
        Uses a greedy assignment ensuring one-to-one matching (a prediction matches at most one GT).

        Args:
            gt_polygons (List[Polygon]): A list of ground truth Shapely Polygon objects.
            pred_polygons (List[Polygon]): A list of predicted Shapely Polygon objects.
            iou_threshold (float, optional): The minimum IoU for a pair to be considered a match.
                                             Defaults to 0.5.

        Returns:
            Tuple[Dict[Tuple[int, int], float], np.ndarray, np.ndarray]: A tuple containing:
                - matched_instances (Dict[(gt_idx, pred_idx), iou]): A dictionary where keys are tuples
                  of matched (ground_truth_index, prediction_index) and values are their IoU scores.
                - gt_matched_map (np.ndarray): A boolean numpy array of shape (num_gt,) indicating
                  whether each ground truth polygon was matched (True) or not (False).
                - pred_matched_map (np.ndarray): A boolean numpy array of shape (num_pred,) indicating
                  whether each predicted polygon was matched (True) or not (False).
        """
        num_gt = len(gt_polygons)
        num_pred = len(pred_polygons)
        iou_matrix = np.zeros((num_gt, num_pred))
        for gt_idx in range(num_gt):
            for pred_idx in range(num_pred):
                iou_matrix[gt_idx, pred_idx] = self.getIOU(gt_polygons[gt_idx], pred_polygons[pred_idx])

        matched_instances = {}
        gt_matched_map = np.zeros(num_gt, dtype=bool)
        pred_matched_map = np.zeros(num_pred, dtype=bool)

        # Greedy matching: Iterate GTs, find best *unmatched* prediction
        for gt_idx in range(num_gt):
            best_iou = -1.0
            best_pred_idx = -1
            for pred_idx in range(num_pred):
                if pred_matched_map[pred_idx]: # Skip already matched predictions
                    continue
                iou = iou_matrix[gt_idx, pred_idx]
                if iou > best_iou and iou >= iou_threshold:
                    best_iou = iou
                    best_pred_idx = pred_idx

            if best_pred_idx != -1:
                # Assign match if found and mark both as matched
                gt_matched_map[gt_idx] = True
                pred_matched_map[best_pred_idx] = True
                matched_instances[(gt_idx, best_pred_idx)] = best_iou

        return matched_instances, gt_matched_map, pred_matched_map

    def compute_pq(self, gt_polygons: List[Polygon], pred_polygons: List[Polygon], iou_threshold=0.5) -> Tuple[float, float, float]:
        """
        Compute Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ).

        Args:
            gt_polygons (List[Polygon]): A list of ground truth Shapely Polygon objects.
            pred_polygons (List[Polygon]): A list of predicted Shapely Polygon objects.
            iou_threshold (float, optional): The IoU threshold used for matching True Positives.
                                             Defaults to 0.5.

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - pq (float): Panoptic Quality score.
                - sq (float): Segmentation Quality score (average IoU of matched instances).
                - rq (float): Recognition Quality score (related to F1 score).
        """
        matched_instances, gt_matched_map, pred_matched_map = self.match_polygons(gt_polygons, pred_polygons, iou_threshold)

        tp = len(matched_instances)
        fp = np.sum(~pred_matched_map) # Unmatched predictions
        fn = np.sum(~gt_matched_map)   # Unmatched ground truths

        # SQ = Average IoU of true positives
        sq = sum(matched_instances.values()) / tp if tp > 0 else 0.0

        # RQ = TP / (TP + 0.5 * FP + 0.5 * FN)
        denominator_rq = tp + 0.5 * fp + 0.5 * fn
        rq = tp / denominator_rq if denominator_rq > 0 else 0.0

        # PQ = SQ * RQ
        pq = sq * rq
        return pq, sq, rq

    def compute_f1(self, gt_polygons: List[Polygon], pred_polygons: List[Polygon], iou_threshold=0.5) -> Tuple[float, float, float]:
        """
        Compute F1 score (F-beta score with beta=1), precision, and recall.

        Args:
            gt_polygons (List[Polygon]): A list of ground truth Shapely Polygon objects.
            pred_polygons (List[Polygon]): A list of predicted Shapely Polygon objects.
            iou_threshold (float, optional): The IoU threshold used for matching True Positives.
                                             Defaults to 0.5.

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - f1 (float): The F1 score.
                - precision (float): The precision score (TP / (TP + FP)).
                - recall (float): The recall score (TP / (TP + FN)).
        """
        # This is equivalent to compute_fbeta with beta=1
        f1, precision, recall = self.compute_fbeta(gt_polygons, pred_polygons, iou_threshold=iou_threshold, beta=1.0)
        return f1, precision, recall

    def compute_fbeta(self, gt_polygons: List[Polygon], pred_polygons: List[Polygon], beta: float, iou_threshold: float = 0.5) -> Tuple[float, float, float]:
        """
        Compute the F-beta score, precision, and recall.

        The F-beta score weights recall more than precision by a factor of beta.
        beta=1.0 is the standard F1 score.
        beta<1.0 weights precision more, beta>1.0 weights recall more.

        Args:
            gt_polygons (List[Polygon]): A list of ground truth Shapely Polygon objects.
            pred_polygons (List[Polygon]): A list of predicted Shapely Polygon objects.
            beta (float): The weight factor for recall. Must be positive.
            iou_threshold (float): The IoU threshold used for matching True Positives. Defaults to 0.5.

        Returns:
            Tuple[float, float, float]: A tuple containing:
                - fbeta (float): The F-beta score.
                - precision (float): The precision score (TP / (TP + FP)).
                - recall (float): The recall score (TP / (TP + FN)).

        Raises:
            ValueError: If beta is not positive.
        """
        if beta <= 0:
            raise ValueError("Beta must be positive.")

        matched_instances, gt_matched_map, pred_matched_map = self.match_polygons(gt_polygons, pred_polygons, iou_threshold)

        tp = len(matched_instances)
        fp = np.sum(~pred_matched_map) # Unmatched predictions
        fn = np.sum(~gt_matched_map)   # Unmatched ground truths

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0 # Or 1.0 if no preds? Standard is usually 0.0
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0 # Or 1.0 if no GTs? Standard is usually 0.0

        # F-beta calculation
        beta_sq = beta**2
        fbeta_denominator = (beta_sq * precision) + recall
        if fbeta_denominator > 0:
            fbeta = (1 + beta_sq) * (precision * recall) / fbeta_denominator
        else:
            fbeta = 0.0 # If precision and recall are both 0

        return fbeta, precision, recall

    def compute_map(self, gt_polygons: List[Polygon], pred_polygons_with_scores: List[Tuple[Polygon, float]], iou_thresholds: Union[List[float], float, None] = None) -> float:
        """
        Compute mean Average Precision (mAP) over a range of IoU thresholds.
        Requires predictions with confidence scores for ranking. Uses all-point interpolation (COCO standard).

        Args:
            gt_polygons (List[Polygon]): A list of ground truth Shapely Polygon objects.
            pred_polygons_with_scores (List[Tuple[Polygon, float]]): A list of tuples, where each
                tuple contains a predicted Shapely Polygon object and its associated confidence score.
            iou_thresholds (Union[List[float], float, None], optional): The IoU threshold(s) over which
                to calculate Average Precision (AP) and then average for mAP.
                - If None, defaults to COCO standard [0.5, 0.55, ..., 0.95].
                - If a float, calculates AP at that single threshold.
                - If a list of floats, calculates AP for each and averages them.
                Defaults to None.

        Returns:
            float: The mean Average Precision (mAP) score. Returns 0.0 if no ground truths or
                   no predictions are provided.
        """
        if not gt_polygons: return 0.0
        if not pred_polygons_with_scores: return 0.0

        # Default IoU thresholds: COCO standard [0.5, 0.55, ..., 0.95]
        if iou_thresholds is None:
            iou_thresholds = np.arange(0.5, 1.0, 0.05).tolist()
        elif isinstance(iou_thresholds, (float, int)):
            iou_thresholds = [float(iou_thresholds)]
        iou_thresholds = np.array(iou_thresholds)

        num_gt = len(gt_polygons)
        average_precisions = []

        # Sort predictions by confidence score (descending)
        sorted_preds = sorted(pred_polygons_with_scores, key=lambda x: x[1], reverse=True)
        pred_polygons_sorted = [p[0] for p in sorted_preds]

        # Calculate AP for each IoU threshold
        for iou_threshold in iou_thresholds:
            tp_list = []  # Stores 1 if prediction is TP, 0 if FP
            gt_matched_map = np.zeros(num_gt, dtype=bool) # Track matched GTs for this IoU threshold

            # Match sorted predictions to GTs
            for pred_idx, pred_polygon in enumerate(pred_polygons_sorted):
                best_iou = -1.0
                best_gt_idx = -1
                for gt_idx, gt_polygon in enumerate(gt_polygons):
                    if gt_matched_map[gt_idx]: # Skip already matched GT
                        continue
                    iou = self.getIOU(gt_polygon, pred_polygon)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                # Assign TP/FP status based on match quality and availability
                if best_iou >= iou_threshold and best_gt_idx != -1 and not gt_matched_map[best_gt_idx]:
                    tp_list.append(1)
                    gt_matched_map[best_gt_idx] = True # Mark GT as matched
                else:
                    tp_list.append(0) # FP

            # Calculate Precision-Recall curve points
            if not tp_list:
                 ap = 0.0
            else:
                tp_list = np.array(tp_list)
                fp_list = 1 - tp_list
                cumulative_tp = np.cumsum(tp_list)
                cumulative_fp = np.cumsum(fp_list)

                recalls = cumulative_tp / num_gt
                precisions = cumulative_tp / (cumulative_tp + cumulative_fp)

                # Calculate AP using All-Point Interpolation (Area under the P-R curve)
                recalls_interp = np.concatenate(([0.0], recalls, [recalls[-1]]))
                precisions_interp = np.concatenate(([0.0], precisions, [0.0]))
                # Make precision monotonically decreasing
                for i in range(len(precisions_interp) - 2, -1, -1):
                    precisions_interp[i] = max(precisions_interp[i], precisions_interp[i+1])
                # Calculate area under curve
                recall_change_indices = np.where(recalls_interp[1:] != recalls_interp[:-1])[0]
                ap = np.sum((recalls_interp[recall_change_indices + 1] - recalls_interp[recall_change_indices]) * precisions_interp[recall_change_indices + 1])

            average_precisions.append(ap)

        # mAP is the mean of APs over the IoU thresholds
        map_score = np.mean(average_precisions) if average_precisions else 0.0
        return map_score
    
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
    
    def compute_fbeta(self, gt_polygons, pred_polygons, beta: float, array_dim=(1024, 1024)) -> tuple:
        """
        Compute the F-beta score, precision, and recall for the given ground truth and predicted polygons.

        The F-beta score weights recall more than precision by a factor of beta.
        beta=1.0 is the standard F1 score.
        beta<1.0 weights precision more, beta>1.0 weights recall more.

        Args:
            gt_polygons (list): List of ground truth polygons.
            pred_polygons (list): List of predicted polygons.
            beta (float): The weight factor for recall. Must be positive.
            array_dim (tuple, optional): Dimensions of the output mask (height, width). Defaults to (1024, 1024).

        Returns:
            tuple: A tuple containing:
                - fbeta (float): The F-beta score.
                - precision (float): The precision score (TP / (TP + FP)).
                - recall (float): The recall score (TP / (TP + FN)).

        Raises:
            ValueError: If beta is not positive.
        """
        if beta <= 0:
            raise ValueError("Beta must be positive.")

        # Create binary masks for ground truth and predictions
        gt_mask = self.polygons_to_mask(gt_polygons, array_dim)
        pred_mask = self.polygons_to_mask(pred_polygons, array_dim)

        # Calculate pixel-level True Positives (TP), False Positives (FP), and False Negatives (FN)
        tp = np.sum((gt_mask == 1) & (pred_mask == 1))
        fp = np.sum((gt_mask == 0) & (pred_mask == 1))
        fn = np.sum((gt_mask == 1) & (pred_mask == 0))

        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0  # if no prediction, precision is considered as 1
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0  # if no ground truth, recall is considered as 1

        # F-beta calculation
        beta_sq = beta**2
        fbeta_denominator = (beta_sq * precision) + recall
        if fbeta_denominator > 0:
            fbeta = (1 + beta_sq) * (precision * recall) / fbeta_denominator
        else:
            fbeta = 0.0  # If precision and recall are both 0

        return fbeta, precision, recall

class RegressionMetrics:
    def __init__(self) -> None:
        """
        Initializes the RegressionMetrics class. No parameters are needed.
        """
        pass

    def compute_rmsle(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Root Mean Squared Logarithmic Error (RMSLE).

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: RMSLE score.
        """
        # Clip negative predictions to 0 before log to avoid errors
        y_pred = np.maximum(y_pred, 0)
        
        # log1p computes log(1 + x)
        log_true = np.log1p(y_true)
        log_pred = np.log1p(y_pred)
        
        return np.sqrt(np.mean(np.square(log_pred - log_true)))

    def compute_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Computes Root Mean Squared Error (RMSE).

        Args:
            y_true (np.ndarray): Array of true values.
            y_pred (np.ndarray): Array of predicted values.

        Returns:
            float: RMSE score.
        """
        return np.sqrt(np.mean(np.square(y_pred - y_true)))
