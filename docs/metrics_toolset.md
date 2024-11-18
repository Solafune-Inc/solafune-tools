# Metrics Toolset

In this repository we also add metrics toolset to accomodate user to be able use metrics we use usually for model training or model evaluation.

## Panoptic Qualities Score Compute Function

This function computes the Panoptic Qualities (PQ) score for a given set of predictions and ground truth. The PQ score is a metric used to evaluate the quality of panoptic segmentation models. The PQ score is computed as the sum of the PQ scores for each class in the dataset. The PQ score for a class is computed as the sum of the true positive, false positive, and false negative values for that class. The PQ score is then normalized by the sum of the true positive and false negative values for that class. The PQ score is a value between 0 and 1, with 1 being the best possible score.

https://arxiv.org/abs/1801.00868

### Authors information

Author: Toru Mitsutake(Solafune) \
Solafune Username: sitoa

### Gettin Started with PQ score

```python
from shapely.geometry import Polygon
from solafune_tools.metrics import PanopticMetric
PQ = PanopticMetric()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [polygon3, polygon4]

pq, sq, rq = PQ.compute_pq(ground_truth_polygons, prediction_polygons)

print("PQ: ", pq)
print("SQ: ", sq)
print("RQ: ", rq)
```

### Input

- ground_truth_polygons: List of polygons representing the ground truth segmentation.
- prediction_polygons: List of polygons representing the predicted segmentation.

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

### Output

- pq: Panoptic Quality score
- sq: Segmentation Quality score
- rq: Recognition Quality score