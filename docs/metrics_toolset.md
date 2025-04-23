# Metrics Toolset

In this repository we also add 🧰 **metrics toolset** to accommodate users in utilizing metrics commonly used for model training or evaluation.

## IOU-Based Metrics

### Panoptic Qualities Score Compute Function 🧮

This function computes the **Panoptic Qualities (PQ)** score using **Intersection Over Union (IoU)** between predicted and ground truth polygons.  
📏 **PQ** is widely used to evaluate **panoptic segmentation** models.

📚 [Paper Implementation](https://arxiv.org/abs/1801.00868)

#### ✍️ Authors information
- **Author**: Toru Mitsutake (Solafune)  
- **Solafune Username**: `sitoa`

---

#### 🚀 Getting Started with PQ Score

##### 🧱 Object Detection Task
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

##### 🧩 Segmentation Task
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

#### 📥 Function Input Parameters
- `ground_truth_polygons`: List of `Polygon` objects representing the ground truth.
- `prediction_polygons`: List of predicted `Polygon` objects.
- `iou_threshold`: Threshold for IoU. **Default: `0.5`**

📘 Polygon type: [`shapely.geometry.Polygon`](https://shapely.readthedocs.io/en/stable/)

#### 📤 Output
- `pq`: **Panoptic Quality** score
- `sq`: **Segmentation Quality** score
- `rq`: **Recognition Quality** score

---

### F1 Score Compute Function 🎯

This function computes the **F1 Score (a.k.a. Dice Coefficient)** using IoU between polygons.  
It is especially useful for **segmentation tasks** and **imbalanced datasets**.

#### ✍️ Authors Information
- **Author**: Lanang Afkaar  
- **Solafune Username**: `Fulankun1412`

---

#### 🚀 Getting Started with F1 Score

##### 🧱 Object Detection Task
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

##### 🧩 Segmentation Task
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

#### 📥 Function Input Parameters
- `ground_truth_polygons`: List of ground truth `Polygon`s
- `prediction_polygons`: List of predicted `Polygon`s
- `iou_threshold`: IoU threshold (default: `0.5`)

#### 📤 Output
- `f1`: **F1 Score**
- `precision`: **Precision**
- `recall`: **Recall**

---

### F-Beta Score Compute Function 🧪

This function computes the **F-Beta Score**, a weighted version of F1 score:

- `beta < 1` ➜ favors **precision**
- `beta > 1` ➜ favors **recall**
- `beta = 1` ➜ equals **F1 score**

---

#### ✍️ Authors Information
- **Author**: Lanang Afkaar  
- **Solafune Username**: `Fulankun1412`

---

#### 🚀 Getting Started with F-Beta Score

##### 🧱 Object Detection Task
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

##### 🧩 Segmentation Task
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

#### 📥 Function Input Parameters
- `ground_truth_polygons`: List of `Polygon` objects
- `prediction_polygons`: List of predicted `Polygon`s
- `iou_threshold`: IoU threshold (default: `0.5`)
- `beta`: Weight for recall (default: `1.0`)

#### 📤 Output
- `f_beta`: **F-Beta Score**
- `precision`: **Precision**
- `recall`: **Recall**

---

### Mean Average Precision Function 🏆

This function computes **Mean Average Precision (mAP)** – a metric used in **object detection** to summarize precision-recall trade-offs.

#### ✍️ Authors Information
- **Author**: Lanang Afkaar  
- **Solafune Username**: `Fulankun1412`

---

#### 🚀 Getting Started with mAP Score

##### 🧱 Object Detection Task

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

##### 🧩 Segmentation Task

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

#### 📥 Function Input Parameters

- `ground_truth_polygons`: List of `Polygon` objects
- `prediction_polygons`: List of `[Polygon, confidence_score]`
- `iou_threshold`: List of thresholds (default: `[0.5, 0.7, 0.95]`)

#### 📤 Output

- `map_score`: **Mean Average Precision** score

---

## Pixel-Based Metrics 🧼

### F1 Score Compute Function 🎯

This function computes **F1 Score** using **pixel-level** evaluation, suitable for high-resolution segmentation comparison.

#### ✍️ Authors Information

- **Author**: Lanang Afkaar  
- **Solafune Username**: `Fulankun1412`

---

#### 🚀 Getting Started with F1 Score

##### 🧱 Object Detection Task

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

##### 🧩 Segmentation Task

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

#### 📥 Function Input Parameters

- `ground_truth_polygons`: List of `Polygon` objects
- `prediction_polygons`: List of predicted `Polygon`s

#### 📤 Output

- `f1`: **F1 Score**
- `precision`: **Precision**
- `recall`: **Recall**