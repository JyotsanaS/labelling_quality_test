def calculate_iou(gt_box, pred_box):
    # Extract coordinates of the bounding boxes
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box

    # Calculate the intersection coordinates
    intersection_x1 = max(gt_x1, pred_x1)
    intersection_y1 = max(gt_y1, pred_y1)
    intersection_x2 = min(gt_x2, pred_x2)
    intersection_y2 = min(gt_y2, pred_y2)

    # Calculate the area of intersection
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(
        0, intersection_y2 - intersection_y1 + 1
    )

    # Calculate the area of each bounding box
    gt_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
    pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    # Calculate the Union area
    union_area = gt_area + pred_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou


def compute_confusion_metrics(
    ground_truth_boxes, predicted_boxes, confidence_threshold, iou_threshold
):
    TP, FP, FN = 0, 0, 0

    used_predictions = set()

    for gt_box in ground_truth_boxes:
        max_confidence_score = 0
        bestMatch = None

        for i, pred_box in enumerate(predicted_boxes):
            if (
                pred_box["confidence"] > confidence_threshold
                and gt_box["label"] == pred_box["label"]
            ):
                iou = calculate_iou(gt_box["bbox"], pred_box["bbox"])

                if iou >= iou_threshold:
                    if pred_box["confidence"] > max_confidence_score:
                        max_confidence_score = pred_box["confidence"]
                        bestMatch = i

        if bestMatch is not None:
            TP += 1
            used_predictions.add(bestMatch)
        else:
            FN += 1

    for i, _ in enumerate(predicted_boxes):
        if i not in used_predictions:
            FP += 1

    return TP, FP, FN


if __name__ == "__main__":
    # Sample use case
    ground_truth_boxes = [
        {"bbox": [0, 0, 50, 50], "label": "cat", "confidence": 1},
        {"bbox": [50, 50, 100, 100], "label": "dog", "confidence": 1},
    ]

    predicted_boxes = [
        {"bbox": [0, 0, 45, 45], "label": "cat", "confidence": 0.9},
        {"bbox": [55, 55, 100, 100], "label": "dog", "confidence": 0.8},
        {"bbox": [10, 10, 40, 40], "label": "cat", "confidence": 0.7},
    ]

    confidence_threshold = 0.6
    alpha = 0.5

    TP, FP, FN = compute_confusion_metrics(
        ground_truth_boxes, predicted_boxes, confidence_threshold, alpha
    )
    print(f"TP: {TP}, FP: {FP}, FN: {FN}")
