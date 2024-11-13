import torch
from intersection_over_union import intersection_over_union
from torchvision import ops
import numpy as np
import matplotlib.pyplot as plt


def mean_average_precision(pred_boxes: list, true_boxes: list, iou_threshold=0.5, format="corners"):
    for threshold in np.arange(iou_threshold, 1, 0.1):
        for class_idx in range(len(pred_boxes)):
            tp = 0
            fp = 0
            total_pos = len(true_boxes[class_idx])

            precision = []
            recall = []

            for pred_box in pred_boxes[class_idx]:
                is_tp = False

                true_box_idx = 0
                while true_box_idx < len(true_boxes[class_idx]) and not is_tp:
                    true_box = true_boxes[class_idx][true_box_idx]
                    iou = intersection_over_union(pred_box, true_box)

                    if iou > iou_threshold:
                        is_tp = True

                    true_box_idx += 1

                tp += is_tp
                fp += not is_tp

                precision.append(tp / (tp + fp))
                recall.append(tp / (tp + total_pos))

        #here we calculate the area under the graph generated, but before i do so and continue with the rest is this so far even the correct way to measure it or am i doing something wrong?

pred_boxes = [
    [  # class 0 dog
        [0, 0, 10, 10],
        [8, 8, 10, 10],
    ],
    [  # class 1 cat
        [5, 5, 10, 10],
        [0, 0, 10, 10],
    ]
]

true_boxes = [
    [
        [0, 0, 9, 9],
        [8, 8, 20, 20],
        [8, 8, 20, 20],
    ],
    [
        [0, 0, 9, 9],
        [8, 8, 20, 20],
    ]
]

mean_average_precision(pred_boxes, true_boxes, )
