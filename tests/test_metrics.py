from shapely.geometry import Polygon
from solafune_tools.metrics import getIOU, PanopticMetric, F1_Metrics

def test_get_iou():
    polygon1 = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
    polygon2 = Polygon([(0, 0), (0, 1), (2, 1), (2, 0)])
    assert getIOU(polygon1, polygon2) == 0.5

def test_compute_pq():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]

    PQ = PanopticMetric()
    
    pq, sq, rq = PQ.compute_pq(true_polygons, pred_polygons)
    assert round(pq,1) == 0.8
    assert sq == 1
    assert round(rq,1) == 0.8
    
def test_compute_pq_same_score_with_dif_order():
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
    PQ = PanopticMetric()
    pq1, sq1, rq1 = PQ.compute_pq([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    pq2, sq2, rq2 = PQ.compute_pq([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert pq1 == pq2
    assert sq1 == sq2
    assert rq1 == rq2

from shapely.geometry import Polygon

def test_compute_pq_zero_score():
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
    PQ = PanopticMetric()
    pq, sq, rq = PQ.compute_pq(true_polygons, pred_polygons)
    assert pq == 0
    assert sq == 0
    assert rq == 0

def test_compute_pq_perfect_score():
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
    PQ = PanopticMetric()
    pq, sq, rq = PQ.compute_pq(true_polygons, pred_polygons)
    assert pq == 1
    assert sq == 1
    assert rq == 1

def test_f1_score():
    polygon1 = Polygon([(1, 2), (2, 4), (3, 1)])
    polygon2 = Polygon([(0, 0), (1, 3), (2, 2), (3, 0)])
    polygon3 = Polygon([(5, 5), (6, 6), (7, 5), (8, 4), (5, 3)])
    polygon4 = Polygon([(2, 2), (3, 4), (4, 4), (5, 2), (3, 1)])
    polygon5 = Polygon([(4, 4), (5, 6), (7, 7), (8, 5), (7, 4)])
    polygon6 = Polygon([(1, 1), (2, 3), (3, 3), (2, 1)])
    polygon7 = Polygon([(3, 3), (4, 5), (6, 5), (7, 3), (5, 2)])
    
    true_polygons = [polygon1, polygon3, polygon5, polygon7]
    pred_polygons = [polygon1, polygon2, polygon3, polygon7]
    
    F1 = F1_Metrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)

    assert round(f1,1) == 0.8
    assert precision == 0.75
    assert round(recall,1) == 0.8

def test_f1_score_same_score_with_dif_order():
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
    F1 = F1_Metrics()
    f1_1, precision_1, recall_1 = F1.compute_f1([true_polygons[i] for i in true_order_1], [pred_polygons[i] for i in pred_order_1])
    f1_2, precision_2, recall_2 = F1.compute_f1([true_polygons[i] for i in true_order_2], [pred_polygons[i] for i in pred_order_2])
    
    assert f1_1 == f1_2
    assert precision_1 == precision_2
    assert recall_1 == recall_2

def test_f1_score_zero_score():
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
    F1 = F1_Metrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)
    assert f1 == 0
    assert precision == 0
    assert recall == 0

def test_f1_score_perfect_score():
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
    F1 = F1_Metrics()
    f1, precision, recall = F1.compute_f1(true_polygons, pred_polygons)
    assert f1 == 1
    assert precision == 1
    assert recall == 1