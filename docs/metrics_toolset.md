# Metrics Toolset

In this repository we also add metrics toolset to accomodate user to be able use metrics we use usually for model training or model evaluation.

## IOU-Based Metrics

### Panoptic Qualities Score Compute Function

This function computes the Panoptic Qualities (PQ) score which leveraged Intersection Over Union (IoU) for a given set of predictions and ground truth polygons. The PQ score is a metric used to evaluate the quality of panoptic segmentation models. The PQ score is computed as the sum of the PQ scores for each class in the dataset. The PQ score for a class is computed as the sum of the true positive, false positive, and false negative values for that class. The PQ score is then normalized by the sum of the true positive and false negative values for that class. The PQ score is a value between 0 and 1, with 1 being the best possible score.

[Paper Implementation](https://arxiv.org/abs/1801.00868)

#### Authors information

Author: Toru Mitsutake(Solafune) \
Solafune Username: sitoa

#### Gettin Started with PQ score

##### Object Detection Task

```python
from solafune_tools.metrics import IOUBasedMetrics, bbox_to_polygon
PQ = IOUBasedMetrics()

bbox1 = (1, 2, 3, 4)
bbox2 = (0, 0, 2, 3)
bbox3 = (5, 5, 7, 6)
bbox4 = (2, 2, 4, 4)

ground_truth_bboxes = [bbox_to_polygon(bbox1), bbox_to_polygon(bbox2)]
prediction_bboxes = [bbox_to_polygon(bbox3), bbox_to_polygon(bbox4)]

pq, sq, rq = PQ.compute_pq(ground_truth_polygons, prediction_polygons, iou_threshold=0.5)

print("PQ: ", pq)
print("SQ: ", sq)
print("RQ: ", rq)
```

##### Segmetation Task

```python
from shapely.geometry import Polygon
from solafune_tools.metrics import IOUBasedMetrics
PQ = IOUBasedMetrics()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [polygon3, polygon4]

pq, sq, rq = PQ.compute_pq(ground_truth_polygons, prediction_polygons, iou_threshold=0.5)

print("PQ: ", pq)
print("SQ: ", sq)
print("RQ: ", rq)
```

#### Function Input Parameters

- ground_truth_polygons: List of polygons representing the ground truth segmentation.
- prediction_polygons: List of polygons representing the predicted segmentation.
- iou_threshold: Threshold for the Intersection over Union. default is 0.5

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

#### Output

- pq: Panoptic Quality score
- sq: Segmentation Quality score
- rq: Recognition Quality score

### F1 Score Compute Function

This function computes the F1 score which leveraged Intersection Over Union (IoU) , specifically tailored for segmentation tasks, to evaluate the quality of predictions against the ground truth polygons. The F1 score, often referred to as the Dice coefficient in segmentation, is a metric that quantifies the overlap between predicted and ground-truth masks. The score ranges from 0 to 1, where 1 indicates a perfect match between predicted and ground-truth masks. This metric is particularly useful for semantic and instance segmentation tasks, especially when handling imbalanced datasets or when high precision and recall are equally critical.

#### Authors Information

Author: Lanang Afkaar \
Solafune username: Fulankun1412

#### Gettin Started with F1 score

##### Object Detection Task

```python
from solafune_tools.metrics import IOUBasedMetrics, bbox_to_polygon
F1_score = IOUBasedMetrics()

bbox1 = (1, 2, 3, 4)
bbox2 = (0, 0, 2, 3)
bbox3 = (5, 5, 7, 6)
bbox4 = (2, 2, 4, 4)

ground_truth_bboxes = [bbox_to_polygon(bbox1), bbox_to_polygon(bbox2)]
prediction_bboxes = [bbox_to_polygon(bbox3), bbox_to_polygon(bbox4)]

f1, precision, recall = F1_score.compute_f1(ground_truth_bboxes, prediction_bboxes, iou_threshold=0.5)

print("F1: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
```

##### Segmetation Task

```python
from shapely.geometry import Polygon
from solafune_tools.metrics import IOUBasedMetrics
F1_score = IOUBasedMetrics()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [polygon3, polygon4]

f1, precision, recall = F1_score.compute_f1(ground_truth_polygons, prediction_polygons, iou_threshold=0.5)

print("F1: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
```

#### Function Input Parameters

- ground_truth_polygons: List of polygons representing the ground truth segmentation.
- prediction_polygons: List of polygons representing the predicted segmentation.
- iou_threshold: Threshold for the Intersection over Union. default is 0.5

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

#### Output

- f1: F1 score
- precision: Precision value
- recall: Recall value

### F-Beta Score Compute Function

This function computes the **F-Beta Score**, which is an extension of the F1 score and is particularly useful for segmentation tasks. The F-Beta score balances precision and recall using a configurable `beta` parameter:

- `beta < 1`: More weight on **precision**
- `beta > 1`: More weight on **recall**
- `beta = 1`: Equivalent to the **F1 score**

The computation is based on the Intersection over Union (IoU) metric between polygons, making it ideal for tasks such as semantic segmentation, instance segmentation, and object detection, especially in cases involving imbalanced datasets.

---

##### Authors Information

Author: Lanang Afkaar \  
Solafune username: Fulankun1412

---

##### Getting Started with F-Beta Score

###### Object Detection Task

```python
from solafune_tools.metrics import IOUBasedMetrics, bbox_to_polygon
F_beta_score = IOUBasedMetrics()

bbox1 = (1, 2, 3, 4)
bbox2 = (0, 0, 2, 3)
bbox3 = (5, 5, 7, 6)
bbox4 = (2, 2, 4, 4)

ground_truth_bboxes = [bbox_to_polygon(bbox1), bbox_to_polygon(bbox2)]
prediction_bboxes = [bbox_to_polygon(bbox3), bbox_to_polygon(bbox4)]

f_beta, precision, recall = F_beta_score.compute_fbeta(
    ground_truth_bboxes, prediction_bboxes, iou_threshold=0.5, beta=0.5
)

print("F-Beta: ", f_beta)
print("Precision: ", precision)
print("Recall: ", recall)
```

##### Segmentation Task

```python
from shapely.geometry import Polygon
from solafune_tools.metrics import IOUBasedMetrics
F_beta_score = IOUBasedMetrics()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [polygon3, polygon4]

f_beta, precision, recall = F_beta_score.compute_fbeta(
    ground_truth_polygons, prediction_polygons, iou_threshold=0.5, beta=2.0
)

print("F-Beta: ", f_beta)
print("Precision: ", precision)
print("Recall: ", recall)
```

---

#### Function Input Parameters

- ground_truth_polygons: List of `shapely.geometry.Polygon` objects representing the ground truth.
- prediction_polygons: List of `shapely.geometry.Polygon` objects representing the predicted segmentation or bounding boxes.
- iou_threshold: Threshold for Intersection over Union (IoU) to consider a prediction as a match. Default is `0.5`.
- beta: The weight of recall in the combined score. Default is `1.0` (F1 score).

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

#### Output

- f_beta: The computed F-Beta score.
- precision`: The precision value based on IoU matches.
- recall: The recall value based on IoU matches.

### Mean Average Precision Function

This function computes the Mean Average Precision (mAP) score, which is a common metric used to evaluate the performance of object detection models. The mAP score is calculated by averaging the precision scores at different recall levels. It provides a single value that summarizes the precision-recall trade-off for a given set of predictions and ground truth annotations. The mAP score ranges from 0 to 1, with 1 indicating perfect precision and recall.

#### Authors Information

Author: Lanang Afkaar \
Solafune username: Fulankun1412

#### Getting Started with mAP score

##### Object Detection Task

```python
from solafune_tools.metrics import IOUBasedMetrics, bbox_to_polygon
mAP = IOUBasedMetrics()

bbox1 = (1, 2, 3, 4)
bbox2 = (0, 0, 2, 3)
bbox3 = (5, 5, 7, 6)
bbox4 = (2, 2, 4, 4)

conf_score3 = 0.6
conf_score4 = 0.87

ground_truth_bboxes = [bbox_to_polygon(bbox1), bbox_to_polygon(bbox2)]
prediction_bboxes = [[bbox_to_polygon(bbox3), conf_score3], [bbox_to_polygon(bbox4), conf_score4]]

map_score = mAP.compute_map(ground_truth_polygons, prediction_polygons, iou_threshold=0.5)

print("mAP: ", map_score)
```

##### Segmetation Task

```python
from shapely.geometry import Polygon
from solafune_tools.metrics import IOUBasedMetrics
mAP = IOUBasedMetrics()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

conf_score3 = 0.66
conf_score4 = 0.77

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [[polygon3, conf_score3], [polygon4, conf_score4]]

map_score = mAP.compute_map(ground_truth_polygons, prediction_polygons, iou_threshold=0.5)

print("mAP: ", map_score)
```

#### Function Input Parameters

- ground_truth_polygons: List of polygons representing the ground truth segmentation.
- prediction_polygons: List of tuple of polygons representing the predicted segmentation and confidence score of the polygon.
- iou_threshold: Threshold for the Intersection over Union. default is [0.5, 0.7, 0.95]

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

#### Output

- map_score: Mean Average Precision score

## Pixel-Based Metrics

### F1 Score Compute Function

This function computes the F1 score which leveraged Pixel-level evaluation, specifically tailored for segmentation tasks, to evaluate the quality of predictions against the ground truth polygons. The F1 score, often referred to as the Dice coefficient in segmentation, is a metric that quantifies the overlap between predicted and ground-truth masks. The score ranges from 0 to 1, where 1 indicates a perfect match between predicted and ground-truth masks. This metric is particularly useful for semantic and instance segmentation tasks, especially when handling imbalanced datasets or when high precision and recall are equally critical.

#### Authors Information

Author: Lanang Afkaar \
Solafune username: Fulankun1412

#### Gettin Started with F1 score

##### Object Detection Task

```python
from solafune_tools.metrics import PixelBasedMetrics, bbox_to_polygon
F1_score = PixelBasedMetrics()

bbox1 = (1, 2, 3, 4)
bbox2 = (0, 0, 2, 3)
bbox3 = (5, 5, 7, 6)
bbox4 = (2, 2, 4, 4)

ground_truth_bboxes = [bbox_to_polygon(bbox1), bbox_to_polygon(bbox2)]
prediction_bboxes = [bbox_to_polygon(bbox3), bbox_to_polygon(bbox4)]

f1, precision, recall = F1_score.compute_f1(ground_truth_bboxes, prediction_bboxes, iou_threshold=0.5)

print("F1: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
```

##### Segmetation Task

```python
from shapely.geometry import Polygon
from solafune_tools.metrics import PixelBasedMetrics
F1_score = PixelBasedMetrics()

polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])

ground_truth_polygons = [polygon1, polygon2]
prediction_polygons = [polygon3, polygon4]

f1, precision, recall = F1_score.compute_f1(ground_truth_polygons, prediction_polygons)

print("F1: ", f1)
print("Precision: ", precision)
print("Recall: ", recall)
```

#### Function Input Parameters

- ground_truth_polygons: List of polygons representing the ground truth segmentation.
- prediction_polygons: List of polygons representing the predicted segmentation.

polygons is shapely.geometry.Polygon object
https://shapely.readthedocs.io/en/stable/

#### Output

- f1: F1 score
- precision: Precision value
- recall: Recall value
