from shapely.geometry import Polygon
from solafune_tools.metrics import IOUBasedMetrics, PixelBasedMetrics, bbox_to_polygon

# ---------------------> The test below is for the IOUBasedMetrics class

## Test bbox_to_polygon
def test_bbox_to_polygon():
    bbox = [0, 0, 1, 1]
    assert bbox_to_polygon(bbox) == Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

## Test for the IOUBasedMetrics of Object Detection Task
def test_get_iou():
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon2 = Polygon([(0, 0), (0, 1), (2, 1), (2, 0)])
    assert IOUBasedMetrics().getIOU(polygon1, polygon2)

def test_iou_based_compute_map_bbox():
    gt_bboxes = [[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1]]
    pred_bboxes = [[[0, 0, 1, 1], 0.7], [[1, 1, 1, 1], 0.8], [[1, 1, 2, 2], 0.9]]
    gt_polygons = [bbox_to_polygon(bbox) for bbox in gt_bboxes]
    pred_polygons = [[bbox_to_polygon(bbox[0]), bbox[1]] for bbox in pred_bboxes]
    
    MAP = IOUBasedMetrics()
    map_score = MAP.compute_map(gt_polygons, pred_polygons)
    assert map_score == 0.4444444444444445

def test_iou_based_compute_map_bbox_same_score_with_dif_order():
    gt_bboxes = [[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1]]
    pred_bboxes = [[[0, 0, 1, 1], 0.6], [[1, 1, 1, 1], 0.5], [[2, 2, 1, 1], 0.6]]
    gt_order_1 = [1, 0, 2]
    pred_order_1 = [0, 1, 2]
    gt_order_2 = [0, 1, 2]
    pred_order_2 = [1, 0, 2]
    gt_polygons = [bbox_to_polygon(gt_bboxes[i]) for i in gt_order_1]
    pred_polygons = [[bbox_to_polygon(pred_bboxes[i][0]), pred_bboxes[i][1]] for i in pred_order_1]
    MAP = IOUBasedMetrics()
    map_score_1 = MAP.compute_map(gt_polygons, pred_polygons)
    gt_polygons = [bbox_to_polygon(gt_bboxes[i]) for i in gt_order_2]
    pred_polygons = [[bbox_to_polygon(pred_bboxes[i][0]), pred_bboxes[i][1]] for i in pred_order_2]
    map_score_2 = MAP.compute_map(gt_polygons, pred_polygons)
    assert map_score_1 == map_score_2

def test_iou_based_compute_map_bbox_zero_score():
    gt_bboxes = [[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1]]
    pred_bboxes = [[[3, 3, 1, 1], 0.6], [[4, 4, 1, 1], 0.6], [[5, 5, 1, 1], 0.6]]
    gt_polygons = [bbox_to_polygon(bbox) for bbox in gt_bboxes]
    pred_polygons = [[bbox_to_polygon(bbox[0]), bbox[1]] for bbox in pred_bboxes]
    MAP = IOUBasedMetrics()
    map_score = MAP.compute_map(gt_polygons, pred_polygons)
    assert map_score == 0

def test_iou_based_compute_map_bbox_perfect_score():
    gt_bboxes = [[0, 0, 1, 1], [1, 1, 1, 1], [2, 2, 1, 1]]
    pred_bboxes = [[[0, 0, 1, 1], 0.7], [[1, 1, 1, 1], 0.7], [[2, 2, 1, 1], 0.7]]
    gt_polygons = [bbox_to_polygon(bbox) for bbox in gt_bboxes]
    pred_polygons = [[bbox_to_polygon(bbox[0]), bbox[1]] for bbox in pred_bboxes]
    MAP = IOUBasedMetrics()
    map_score = MAP.compute_map(gt_polygons, pred_polygons, iou_thresholds=0.5)
    assert map_score == 1

## Test for the IOUBasedMetrics of Segmentation Task
def test_get_iou():
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon2 = Polygon([(0, 0), (0, 1), (2, 1), (2, 0)])
    assert IOUBasedMetrics().getIOU(polygon1, polygon2) == 0.5

### Test for the IOUBasedMetrics of Segmentation Task Compute PQ
def test_iou_based_compute_pq_segmentation():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]

    PQ = IOUBasedMetrics()
    
    pq, sq, rq = PQ.compute_pq(true_polygons, pred_polygons)
    assert round(pq,1) == 0.8
    assert sq == 1
    assert round(rq,1) == 0.8
    
def test_iou_based_compute_pq_segmentation_same_score_with_dif_order():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(7, 3), (8, 4), (9, 3), (10, 2), (7, 1), (7, 3)]),
        Polygon([(9, 8), (10, 10), (12, 11), (13, 9), (12, 8), (9, 8)]),
        Polygon([(4, 4), (5, 6), (7, 6), (8, 4), (6, 3), (4, 4)]),
    ]
    true_order_1 = [1, 0, 2]
    pred_order_1 = [0, 1, 2]
    true_order_2 = [0, 1, 2]
    pred_order_2 = [1, 0, 2]
    PQ = IOUBasedMetrics()
    pq1, sq1, rq1 = PQ.compute_pq([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    pq2, sq2, rq2 = PQ.compute_pq([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert round(pq1, 1) == round(pq2, 1)
    assert round(sq1, 1) == round(sq2, 1)
    assert round(rq1, 1) == round(rq2, 1)

def test_compute_pq_segmentation_zero_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(20, 20), (21, 21), (22, 20), (23, 19), (20, 18), (20, 20)]),
        Polygon([(30, 30), (31, 32), (33, 33), (34, 31), (33, 30), (30, 30)]),
        Polygon([(40, 40), (41, 42), (43, 42), (44, 40), (42, 39), (40, 40)]),
    ]
    PQ = IOUBasedMetrics()
    pq, sq, rq = PQ.compute_pq(true_polygons, pred_polygons)
    assert pq == 0
    assert sq == 0
    assert rq == 0

def test_iou_based_compute_pq_segmentation_perfect_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    PQ = IOUBasedMetrics()
    pq, sq, rq = PQ.compute_pq(true_polygons, pred_polygons)
    assert pq == 1
    assert sq == 1
    assert rq == 1

### Test for the IOUBasedMetrics of Segmentation F1 Score
def test_iou_based_f1_score_segmentation():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]
    
    F1 = IOUBasedMetrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)

    assert round(f1,1) == 0.8
    assert precision == 0.75
    assert round(recall,1) == 0.8

def test_iou_based_f1_score_segmentation_same_score_with_dif_order():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(7, -3), (8, -2), (9, -3), (10, -4), (7, -5), (7, -3)]),
        Polygon([(9, 8), (10, 10), (12, 11), (13, 9), (12, 8), (9, 8)]),
        Polygon([(4, 4), (5, 6), (7, 6), (8, 4), (6, 3), (4, 4)]),
    ]
    true_order_1 = [1, 0, 2]
    pred_order_1 = [0, 1, 2]
    true_order_2 = [0, 1, 2]
    pred_order_2 = [1, 0, 2]
    F1 = IOUBasedMetrics()
    f1_1, precision_1, recall_1 = F1.compute_f1([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    f1_2, precision_2, recall_2 = F1.compute_f1([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert f1_1 == f1_2
    assert precision_1 == precision_2
    assert recall_1 == recall_2

def test_iou_based_f1_score_segmentation_zero_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(20, 20), (21, 21), (22, 20), (23, 19), (20, 18), (20, 20)]),
        Polygon([(30, 30), (31, 32), (33, 33), (34, 31), (33, 30), (30, 30)]),
        Polygon([(40, 40), (41, 42), (43, 42), (44, 40), (42, 39), (40, 40)]),
    ]
    F1 = IOUBasedMetrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)
    assert f1 == 0
    assert precision == 0
    assert recall == 0

def test_iou_based_f1_score_segmentation_perfect_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    F1 = IOUBasedMetrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)
    assert f1 == 1
    assert precision == 1
    assert recall == 1

def test_iou_based_fbeta_score_segmentation():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5]
    pred_polygons = [polygon1, polygon2]
    
    Fbeta = IOUBasedMetrics()
    fbeta, _, _ = Fbeta.compute_fbeta(true_polygons, pred_polygons, beta=1)
    
    assert round(fbeta,1) == 0.4

def test_iou_based_fbeta_score_segmentation_same_score_with_dif_order():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(7, -3), (8, -2), (9, -3), (10, -4), (7, -5), (7, -3)]),
        Polygon([(9, 8), (10, 10), (12, 11), (13, 9), (12, 8), (9, 8)]),
        Polygon([(4, 4), (5, 6), (7, 6), (8, 4), (6, 3), (4, 4)]),
    ]
    true_order_1 = [1, 0]
    pred_order_1 = [0]
    true_order_2 = [0]
    pred_order_2 = [1]
    Fbeta = IOUBasedMetrics()
    fbeta_score_1 = Fbeta.compute_fbeta([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1], beta=1)
    fbeta_score_2 = Fbeta.compute_fbeta([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2], beta=1)
    
    assert math.isclose(fbeta_score_1, fbeta_score_2, rel_tol=1e-9)

def test_iou_based_fbeta_score_segmentation_zero_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(20, 20), (21, 21), (22, 20), (23, 19), (20, 18), (20, 20)]),
        Polygon([(30, 30), (31, 32), (33, 33), (34, 31), (33, 30), (30, 30)]),
        Polygon([(40, 40), (41, 42), (43, 42), (44, 40), (42, 39), (40, 40)]),
    ]
    Fbeta = IOUBasedMetrics()
    fbeta_score, _, _ = Fbeta.compute_fbeta(true_polygons, pred_polygons, beta=1)
    assert fbeta_score == 0.0

def test_iou_based_fbeta_score_segmentation_perfect_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    Fbeta = IOUBasedMetrics()
    fbeta_score, _, _ = Fbeta.compute_fbeta(true_polygons, pred_polygons, beta=1)
    assert fbeta_score == 1.0

### Test for the IOUBasedMetrics of Segmentation Mean Average Precision
def test_iou_based_compute_map_segmentation():
    polygon1 = [Polygon([(1, 2), (2, 4), (3, 1)]), 1.0 ]
    polygon2 = [Polygon([(0, 0), (1, 3), (2, 2), (3, 0)]), 1.0 ]
    polygon3 = [Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)]), 1.0 ]
    polygon4 = [Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)]), 1.0 ]
    polygon5 = [Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)]), 1.0 ]
    polygon6 = [Polygon([(1, 1), (2, 3), (3, 3), (2, 1)]), 0.75 ]
    polygon7 = [Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)]), 1.0]
    
    true_polygons = [polygon1[0], polygon3[0], polygon5[0], polygon7[0]] 
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]
    
    MAP = IOUBasedMetrics()
    map_score = MAP.compute_map(true_polygons, pred_polygons)
    assert round(map_score,1) == 0.6

def test_iou_based_compute_map_segmentation_same_score_with_dif_order():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        [Polygon([(7, -3), (8, -2), (9, -3), (10, -4), (7, -5), (7, -3)]), 1.0],
        [Polygon([(9, 8), (10, 10), (12, 11), (13, 9), (12, 8), (9, 8)]), 0.8],
        [Polygon([(4, 4), (5, 6), (7, 6), (8, 4), (6, 3), (4, 4)]), 0.9],
    ]
    true_order_1 = [1, 0, 2]
    pred_order_1 = [0, 1, 2]
    true_order_2 = [0, 1, 2]
    pred_order_2 = [1, 0, 2]
    MAP = IOUBasedMetrics()
    map_score_1 = MAP.compute_map([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    map_score_2 = MAP.compute_map([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert map_score_1 == map_score_2

def test_iou_based_compute_map_segmentation_zero_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        [Polygon([(20, 20), (21, 21), (22, 20), (23, 19), (20, 18), (20, 20)]), 0.5],
        [Polygon([(30, 30), (31, 32), (33, 33), (34, 31), (33, 30), (30, 30)]), 0.7],
        [Polygon([(40, 40), (41, 42), (43, 42), (44, 40), (42, 39), (40, 40)]), 0.6],
    ]
    MAP = IOUBasedMetrics()
    map_score = MAP.compute_map(true_polygons, pred_polygons)
    assert map_score == 0

def test_iou_based_compute_map_segmentation_perfect_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        [Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]), 0.8],
        [Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]), 0.9],
        [Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]), 0.6],
    ]
    MAP = IOUBasedMetrics()
    map_score = MAP.compute_map(true_polygons, pred_polygons, iou_thresholds=0.5)
    assert map_score == 1

# ---------------------> Below are the tests for PixelBasedMetrics

def test_pixel_based_polygon_to_mask():
    polygon = [Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])]
    mask = PixelBasedMetrics().polygons_to_mask(polygon, (2, 2))
    assert mask.tolist() == [[1, 0], [0, 0]]

def test_pixel_based_compute_f1():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]
    
    F1 = PixelBasedMetrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)

    assert round(f1,1) == 0.8
    assert round(precision, 1) == 0.8
    assert round(recall,1) == 0.8

def test_pixel_based_f1_score_same_score_with_dif_order():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(7, -3), (8, -2), (9, -3), (10, -4), (7, -5), (7, -3)]),
        Polygon([(9, 8), (10, 10), (12, 11), (13, 9), (12, 8), (9, 8)]),
        Polygon([(4, 4), (5, 6), (7, 6), (8, 4), (6, 3), (4, 4)]),
    ]
    true_order_1 = [1, 0, 2]
    pred_order_1 = [0, 1, 2]
    true_order_2 = [0, 1, 2]
    pred_order_2 = [1, 0, 2]
    F1 = PixelBasedMetrics()
    f1_1, precision_1, recall_1 = F1.compute_f1([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    f1_2, precision_2, recall_2 = F1.compute_f1([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert f1_1 == f1_2
    assert precision_1 == precision_2
    assert recall_1 == recall_2

def test_pixel_based_f1_score_zero_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(20, 20), (21, 21), (22, 20), (23, 19), (20, 18), (20, 20)]),
        Polygon([(30, 30), (31, 32), (33, 33), (34, 31), (33, 30), (30, 30)]),
        Polygon([(40, 40), (41, 42), (43, 42), (44, 40), (42, 39), (40, 40)]),
    ]
    F1 = PixelBasedMetrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)
    assert f1 == 0
    assert precision == 0
    assert recall == 0

def test_pixel_based_f1_score_perfect_score():
    true_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    pred_polygons = [
        Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3), (5, 5)]),
        Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4), (4, 4)]),
        Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2), (3, 3)]),
    ]
    F1 = PixelBasedMetrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)
    assert f1 == 1
    assert precision == 1
    assert recall == 1

