import numpy as np
import torch

def intersection_over_union(boxes_predict: torch.Tensor, boxes_target: torch.Tensor, format="corners", epsilon=1e-8) -> torch.Tensor:
    if type(boxes_predict) == list:
        boxes_predict = torch.tensor(boxes_predict).float()
    if type(boxes_target) == list:
        boxes_target = torch.tensor(boxes_target).float()


    if format == "midpoint":
        # Convert (x_center, y_center, width, height) to (x1, y1, x2, y2)
        box1_x1 = boxes_predict[..., 0:1] - boxes_predict[..., 2:3] / 2
        box1_y1 = boxes_predict[..., 1:2] - boxes_predict[..., 3:4] / 2
        box1_x2 = boxes_predict[..., 0:1] + boxes_predict[..., 2:3] / 2
        box1_y2 = boxes_predict[..., 1:2] + boxes_predict[..., 3:4] / 2

        box2_x1 = boxes_target[..., 0:1] - boxes_target[..., 2:3] / 2
        box2_y1 = boxes_target[..., 1:2] - boxes_target[..., 3:4] / 2
        box2_x2 = boxes_target[..., 0:1] + boxes_target[..., 2:3] / 2
        box2_y2 = boxes_target[..., 1:2] + boxes_target[..., 3:4] / 2

    elif format == "xywh":
        # Convert (x1, y1, width, height) to (x1, y1, x2, y2)
        box1_x1 = boxes_predict[..., 0:1]
        box1_y1 = boxes_predict[..., 1:2]
        box1_x2 = boxes_predict[..., 0:1] + boxes_predict[..., 2:3]
        box1_y2 = boxes_predict[..., 1:2] + boxes_predict[..., 3:4]

        box2_x1 = boxes_target[..., 0:1]
        box2_y1 = boxes_target[..., 1:2]
        box2_x2 = boxes_target[..., 0:1] + boxes_target[..., 2:3]
        box2_y2 = boxes_target[..., 1:2] + boxes_target[..., 3:4]

    elif format == "corners":
        # Already in (x1, y1, x2, y2) format
        box1_x1 = boxes_predict[..., 0:1]
        box1_y1 = boxes_predict[..., 1:2]
        box1_x2 = boxes_predict[..., 2:3]
        box1_y2 = boxes_predict[..., 3:4]

        box2_x1 = boxes_target[..., 0:1]
        box2_y1 = boxes_target[..., 1:2]
        box2_x2 = boxes_target[..., 2:3]
        box2_y2 = boxes_target[..., 3:4]
    else:
        raise ValueError(f"Unknown format: {format}. Supported formats are 'corners', 'midpoint', 'xywh'.")

    # Calculate the intersection coordinates
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Calculate intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Calculate the area of each box
    box1_area = (box1_x2 - box1_x1).clamp(0) * (box1_y2 - box1_y1).clamp(0)
    box2_area = (box2_x2 - box2_x1).clamp(0) * (box2_y2 - box2_y1).clamp(0)

    # Union area
    union = box1_area + box2_area - intersection

    # IoU calculation
    return intersection / (union + epsilon)



predicted = torch.tensor([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
])

target = torch.tensor([
    [ 5, 5, 6, 6],
    [20, 20, 35, 35],
])

print(intersection_over_union(predicted, target))