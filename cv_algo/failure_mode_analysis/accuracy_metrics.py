from shapely.geometry import Polygon

def calculate_iou(gt_box, pred_box):
     """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Parameters:
    gt_box (tuple): Tuple containing the coordinates of the ground truth bounding box in the format (x1, y1, x2, y2).
    pred_box (tuple): Tuple containing the coordinates of the predicted bounding box in the same format as gt_box.

    Returns:
    float: The IoU value
    """
    # Extract coordinates of the bounding boxes
    gt_x1, gt_y1, gt_x2, gt_y2 = gt_box
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_box

    # Calculate the intersection coordinates
    intersection_x1 = max(gt_x1, pred_x1)
    intersection_y1 = max(gt_y1, pred_y1)
    intersection_x2 = min(gt_x2, pred_x2)
    intersection_y2 = min(gt_y2, pred_y2)

    # Calculate the area of intersection
    intersection_area = max(0, intersection_x2 - intersection_x1 + 1) * max(0, intersection_y2 - intersection_y1 + 1)

    # Calculate the area of each bounding box
    gt_area = (gt_x2 - gt_x1 + 1) * (gt_y2 - gt_y1 + 1)
    pred_area = (pred_x2 - pred_x1 + 1) * (pred_y2 - pred_y1 + 1)

    # Calculate the Union area
    union_area = gt_area + pred_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def compute_confusion_metrics(ground_truth_boxes, predicted_boxes, confidence_threshold, iou_threshold):
    """
    Compute confusion metrics including True Positives (TP), False Positives (FP), and False Negatives (FN) based on ground truth and predicted bounding boxes.

    Parameters:
    ground_truth_boxes (list of dicts): List containing dictionaries representing ground truth bounding boxes.
    predicted_boxes (list of dicts): List containing dictionaries representing predicted bounding boxes.
    confidence_threshold (float): Confidence threshold for predicted bounding boxes.
    iou_threshold (float): IoU threshold for matching ground truth and predicted bounding boxes.

    Returns:
    tuple: A tuple containing:
           - TP (True Positives)
           - FP (False Positives)
           - FN (False Negatives)
    """
    TP, FP, FN = 0, 0, 0

    used_predictions = set()

    for gt_box in ground_truth_boxes:
        max_confidence_score = 0
        bestMatch = None

        for i, pred_box in enumerate(predicted_boxes):
            if pred_box['score'] > confidence_threshold and gt_box['label'] == pred_box['category_id']:
                iou = calculate_iou(gt_box['bbox'], pred_box['bbox'])
                if iou >= iou_threshold:
                    if pred_box['score'] > max_confidence_score:
                        max_confidence_score = pred_box['score']
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

def accuracy_metrics(TP, FP, FN):
    """
    Compute accuracy metrics.

    Args:
        TP (int): True Positives.
        FP (int): False Positives.
        FN (int): False Negatives.

    Returns:
        list: Accuracy, Precision, Recall, F1 score.
    """
    acc = TP / (TP+FP+FN)
    precision = TP / (TP+FP)
    recall = TP / (TP+FN)
    f1 = (2*precision*recall) / (precision+recall) if TP!=0 else 0
    return [acc, precision, recall, f1]

def display_results(Acc, Precision, Recall, F1):
    """
    Display evaluation results.

    Args:
        Acc (list): List of accuracy values.
        Precision (list): List of precision values.
        Recall (list): List of recall values.
        F1 (list): List of F1 scores.
    """
    print("Accuracy: \t", sum(Acc)/len(Acc))
    print("Precision: \t", sum(Precision)/len(Precision))
    print("Recall: \t", sum(Recall)/len(Recall))
    print("F1: \t\t", sum(F1)/len(F1))

def calculate_iou_poly(poly1, poly2):
    """
    Calculate Intersection over Union (IoU) between two polygons.
    
    Parameters:
    poly1, poly2: Shapely Polygon objects representing two polygons.
    
    Returns:
    IoU: Intersection over Union between poly1 and poly2.
    """
    if(poly1.is_valid == False): poly1 = poly1.buffer(0)
    if(poly2.is_valid == False): poly2 = poly2.buffer(0)
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    iou = intersection / union if union != 0 else 0
    return iou

def calculate_dice(poly1, poly2):
    """
    Calculate Dice coefficient between two polygons.
    
    Args:
    poly1, poly2: Shapely Polygon objects representing two polygons.
    
    Returns:
    dice (float): dice between poly1 and poly2.
    """
    if(poly1.is_valid == False): poly1 = poly1.buffer(0)
    if(poly2.is_valid == False): poly2 = poly2.buffer(0)
    intersection = poly1.intersection(poly2).area
    dice = (2 * intersection) / (poly1.area + poly2.area)
    return dice

def convert_list_to_polygon(label_list):
    """
    Convert a list of dictionaries with polygon coordinates to a list of dictionaries with Shapely Polygon objects.

    Parameters:
    label_list (list): A list of dictionaries where each dictionary contains a 'polygon' key with a list of tuples
                       representing the coordinates of a polygon.

    Returns:
    list: A list of dictionaries where each 'polygon' value is a Shapely Polygon object.
    """
    ans = []
    for dic in label_list:
        dic['polygon'] = Polygon(tuple(dic['polygon']))
        ans.append(dic)
    return ans

def calculate_polygon_iou(gt_polygons, pred_polygons, label_idx, iou_threshold, confidence_threshold):
    """
    Calculate the total Intersection over Union (IoU) and Dice coefficients for polygons with a specific label index.

    Parameters:
    ground_truth_polygons (list): A list of dictionaries representing ground truth polygons.
    predicted_polygons (list): A list of dictionaries representing predicted polygons.
    label_idx (int): The label index for which to calculate IoU and Dice coefficients.
    iou_threshold (float): The threshold for the IoU coefficient.
    confidence_threshold (float): The threshold for the confidence.

    Returns:
    iou - The total IoU score for polygons with the specified label index.
    iou_count - The number of polygons with the specified label index.
    """
    iou_list = []
    ground_truth_polygons  = convert_list_to_polygon(gt_polygons)
    predicted_polygons = convert_list_to_polygon(pred_polygons) 
    for gt_poly in ground_truth_polygons:
        max_confidence_score, max_iou_score = 0, 0
        if gt_poly['label'] == label_idx:
            for i, pred_poly in enumerate(predicted_polygons):
                if pred_poly['score'] > confidence_threshold and pred_poly['label'] == label_idx:
                    iou = calculate_iou_poly(gt_poly['polygon'], pred_poly['polygon'])
                    if iou >= iou_threshold and pred_poly['score'] > max_confidence_score:
                        max_confidence_score = pred_poly['score']
                        max_iou_score = iou
            iou_list.append(max_iou_score)
    iou = sum(iou_list)
    return iou, len(iou_list)

def calculate_polygon_dice(ground_truth_polygons, predicted_polygons, label_idx, dice_threshold, confidence_threshold):
    """
    Calculate the total Intersection over Union (IoU) and Dice coefficients for polygons with a specific label index.

    Parameters:
    ground_truth_polygons (list): A list of dictionaries representing ground truth polygons.
    predicted_polygons (list): A list of dictionaries representing predicted polygons.
    label_idx (int): The label index for which to calculate IoU and Dice coefficients.
    dice_threshold (float): The threshold for the Dice coefficient.
    confidence_threshold (float): The threshold for the confidence.

    Returns:
    dice - The total Dice coefficient score for polygons with the specified label index.
    dice_count - The number of polygons with the specified label index.
    """
    dice_list = []

    for gt_poly in ground_truth_polygons:
        max_confidence_score, max_dice_score = 0, 0
        if gt_poly['label'] == label_idx:
            for i, pred_poly in enumerate(predicted_polygons):
                if pred_poly['score'] > confidence_threshold and pred_poly['label'] == label_idx:
                    dice = calculate_dice(gt_poly['polygon'], pred_poly['polygon'])
                    if dice >= dice_threshold and pred_poly['score'] > max_confidence_score:
                        max_confidence_score = pred_poly['score']
                        max_dice_score = dice
            dice_list.append(max_dice_score)
    dice = sum(dice_list)
    return dice, len(dice_list)
