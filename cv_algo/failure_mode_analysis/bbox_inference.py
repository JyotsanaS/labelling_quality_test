import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from accuracy_metrics import calculate_iou, compute_confusion_metrics, accuracy_metrics, display_results
from tqdm import tqdm
import json

def read_gt(file_path, image_path, isAddOne):
    """
    Read ground truth bounding boxes from a file.

    Args:
        file_path (str): Path to the file containing ground truth bounding boxes.
        image_path (str): Path to the image corresponding to the bounding boxes.

    Returns:
        bounding_boxes (list): List of ground truth bounding boxes.
    """
    bounding_boxes = []
    # isAddOne = file_path.split('/')[-4] == "hardtail_cv_Feburary_2023_set2-1"
    # Read the image to get its height and width
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    img_id = image_path.split('/')[-1].replace('.jpg', '')
    with open(file_path, 'r') as file:
        for line in file:
            # print(line)
            values = line.strip().split()
            obj_class = int(values[0]+1) if isAddOne else int(values[0]+1)
            segmentation = list(map(float, values[1:]))
            x_values = segmentation[::2]  # Extract every other element starting from index 0
            y_values = segmentation[1::2]  # Extract every other element starting from index 1
            x_min = min(x_values)
            x_max = max(x_values)
            y_min = min(y_values)
            y_max = max(y_values)
            # Normalize bounding box coordinates with respect to image size
            x_min_normalized = x_min * width
            x_max_normalized = x_max * width
            y_min_normalized = y_min * height
            y_max_normalized = y_max * height
            bounding_box = {
                'image_id': img_id,
                'label': obj_class,
                'confidence': 1,
                'bbox' : [x_min_normalized, y_min_normalized, x_max_normalized, y_max_normalized]
            }
            # Append the bounding box to the list
            bounding_boxes.append(bounding_box)
    return bounding_boxes

def read_model_output(file_path, image_path, isAddOne):
    """
    Read predicted bounding boxes from a file.

    Args:
        file_path (str): Path to the file containing predicted bounding boxes.
        image_path (str): Path to the image corresponding to the bounding boxes.

    Returns:
        bounding_boxes (list): List of predicted bounding boxes.
    """
    # Initialize an empty list to store bounding boxes
    bounding_boxes = []
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    img_id = image_path.split('/')[-1].replace('.jpg', '')

    # Read the file line by line
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            # Split the line into components
            components = line.strip().split()

            # Extract relevant information
            obj_class = int(components[0]+1) if isAddOne else int(components[0])
            x_center = float(components[1])
            y_center = float(components[2])
            width_ratio = float(components[3])
            height_ratio = float(components[4])
            confidence = float(components[5])

            # Compute bounding box coordinates
            x_min = max(0, int((x_center - 0.5 * width_ratio) * width))
            y_min = max(0, int((y_center - 0.5 * height_ratio) * height))
            x_max = min(width, int((x_center + 0.5 * width_ratio) * width))
            y_max = min(height, int((y_center + 0.5 * height_ratio) * height))
            
            bounding_box = {
                "id": img_id,
                "image_id": img_id,
                "category_id": obj_class,
                "score": confidence,
                "bbox" : [x_min, y_min, x_max, y_max],
                "iscrowd": 0,
            }

            # Append the bounding box to the list
            bounding_boxes.append(bounding_box)

    return bounding_boxes

def storeDataFrame():
    """
    Store ground truth and predicted bounding box labels from different folders into combined folders.
    """
    main_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/training_valid_dataset/"
    prediction_folder_path = "/home/pratham/walmart_poc/yolov7/seg/runs/predict-seg/"
    combine_folder_names = ["hardtail_cv_april_2023-4", "hardtail_cv_february_2023-13", "hardtail_cv_Feburary_2023_set2-1"]
    combined_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/combined_valid_bbox/" 
    for folder_name in combine_folder_names:
        folder_path = main_folder_path + folder_name
        images_path = folder_path + "/valid/images/"
        labels_path = folder_path + "/valid/labels/"
        pred_labels_path = prediction_folder_path + folder_name + "__valid/labels/"
        isAddOne = True if folder_name == "hardtail_cv_Feburary_2023_set2-1" else False
        for label_name in sorted(os.listdir(labels_path)):
            label_path = labels_path + label_name
            image_path = images_path + label_name.replace('.txt', '.jpg')
            label_dic = read_gt(label_path, image_path, isAddOne)
            json_object = json.dumps(label_dic)
            with open(combined_folder_path + 'ground_truth_labels/' + label_name, "w") as outfile:
                outfile.write(json_object)
            pred_label_path = pred_labels_path + label_name
            pred_label_dic = read_model_output(pred_label_path, image_path, isAddOne)
            json_object = json.dumps(pred_label_dic)
            with open(combined_folder_path + 'predicted_labels/' + label_name, "w") as outfile:
                outfile.write(json_object)   

def top_6_elements(matrix):
    """
    Find the top 6 elements in a 2D matrix along with their indices, and return the range of rows and columns containing these elements.

    Parameters:
    matrix (list of lists): The input 2D matrix containing numerical elements.

    Returns:
    tuple: A tuple containing four elements:
           - The minimum row index of the top 6 elements.
           - The maximum row index of the top 6 elements.
           - The minimum column index of the top 6 elements.
           - The maximum column index of the top 6 elements.
    """
    elements_with_indices = []
    for i, row in enumerate(matrix):
        for j, element in enumerate(row):
            elements_with_indices.append((element, (i, j)))
    
    top_6 = sorted(elements_with_indices, key=lambda x: x[0], reverse=True)[:6]

    print("Top 6 elements and their indices:")
    for i, (value, index) in enumerate(top_6, start=1):
        print(f"{i}. Element: {value}, Index: {index}")

    min_row = min(row for _, (row, _) in top_6)
    max_row = max(row for _, (row, _) in top_6)
    min_col = min(col for _, (_, col) in top_6)
    max_col = max(col for _, (_, col) in top_6)
    return 0.95-max_row*0.05, 0.95-min_row*0.05, min_col*0.05, max_col*0.05 #As the axis for plotting y axis(IoU) threshold is reversed

if __name__ == "__main__":
    # # Use storeDataFrame only once to get store ground truths and prediction in combined folder
    # storeDataFrame()
    
    # Define paths and parameters
    labels_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/combined_valid_bbox/ground_truth_labels/"
    predictions_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/combined_valid_bbox/predicted_labels/"

    confidence_threshold = 0.0
    iou_threshold = 0.0

    ans = np.zeros((20, 20))
    x_axis = [round(value, 2) for value in np.arange(0, 1, 0.05)]
    for i in tqdm(range(0, 20)):
        for j in range(0,20):
            iou_threshold = i * 0.05
            confidence_threshold = j * 0.05
            # Lists to store evaluation metrics
            Acc, Precision, Recall, F1 = [],[],[],[]

            for label_name in sorted(os.listdir(labels_folder_path)):
                file_path_gt = labels_folder_path + label_name
                file_path_pred = predictions_folder_path + label_name

                # Load ground truth and predicted bounding boxes
                ground_truth_bboxs_string = open(file_path_gt).read()
                ground_truth_bboxs = json.loads(ground_truth_bboxs_string)
                predicted_bboxs_string = open(file_path_pred).read()
                predicted_bboxs = json.loads(predicted_bboxs_string)
                # # Compute confusion matrix metrics
                TP, FP, FN = compute_confusion_metrics(ground_truth_bboxs, predicted_bboxs, confidence_threshold, iou_threshold)

                # Compute accuracy metrics
                acc, precision, recall, f1 = accuracy_metrics(TP, FP, FN)

                # Append to lists
                Acc.append(acc)
                Precision.append(precision)
                Recall.append(recall)
                F1.append(f1)
            ans[i][j] = round(sum(F1)/len(F1), 3)
    ans = ans[::-1]
    plt.rcParams["figure.figsize"] = (10, 10)
    fig, ax = plt.subplots()
    im = ax.imshow(ans, interpolation='none')

    # Set x and y ticks with labels
    ax.set_xticks(np.arange(len(x_axis)), labels=x_axis)
    ax.set_yticks(np.arange(len(x_axis)), labels=x_axis[::-1])

    # Add text annotations to the plot
    for i in range(0, 20):
        for j in range(0, 20):
            text = ax.text(i, j, round(ans[j][i], 1), ha="center", va="center", color="w")

    min_row, max_row, min_col, max_col = top_6_elements(ans)
    print("Range of IoU (rows): ", min_row," : ", max_row )
    print("Range of Confidence (cols): ", min_col, " : ",max_col)

    # Set title and labels
    plt.title("F1 Score")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("IoU Threshold")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('F1 Score')
    plt.savefig('plot.png')
