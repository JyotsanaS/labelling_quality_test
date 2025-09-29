import os
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from accuracy_metrics import calculate_polygon_iou, calculate_polygon_dice
import json
from tqdm import tqdm

def convert_string_to_tuples(label_path, isAddOne, isPredicted):
    """
    Convert labeled polygons from a file to a list of dictionaries.

    Parameters:
    label_path (str): File path to the labeled polygons file.
    isAddOne (int): Increment label values by 1 if set to 1, else unchanged.
    isPredicted (bool): Include prediction confidence scores if True.

    Returns:
    list: List of dictionaries with 'label' and 'polygon' keys.
          'label': Category id of the polygon
          'polygon': List of tuples representing polygon vertices.
          (Optional) 'score': Confidence score if isPredicted is True.
    """
    polys = open(label_path).readlines()
    formated_polys = []
    for poly in polys:
        ans = {}
        poly = poly[:-1] #remove '/n' from last coordinate
        poly = [float(num) for num in poly.split()]
        ans['label'] = poly[0] + isAddOne
        if(isPredicted): 
            ans['score'] = poly[-1] # confidence score
        poly = poly[1: -1] # Remove the label from the xy pairs

        # Create tuples from consecutive pairs of co-ordinates
        tuples_list = [(poly[i], poly[i+1]) for i in range(0, len(poly)-1, 2)]
        ans['polygon'] = tuples_list
        formated_polys.append(ans)
    return formated_polys

def storeDataFrame():
    """
    Store ground truth and predicted labels from different folders into combined folders.
    """
    main_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/training_valid_dataset/"
    prediction_folder_path = "/home/pratham/walmart_poc/yolov7/seg/runs/predict-seg/"
    combine_folder_names = ["hardtail_cv_april_2023-4", "hardtail_cv_february_2023-13", "hardtail_cv_Feburary_2023_set2-1"]
    combined_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/combined_valid_segment/" 
    for folder_name in combine_folder_names:
        folder_path = main_folder_path + folder_name
        labels_path = folder_path + "/valid/labels/"
        pred_labels_path = prediction_folder_path + folder_name + "__poly/labels/"
        isAddOne = 1 if folder_name == "hardtail_cv_Feburary_2023_set2-1" else 0
        for label_name in sorted(os.listdir(labels_path)):
            label_path = labels_path + label_name
            label_dic = convert_string_to_tuples(label_path, isAddOne, False)
            json_object = json.dumps(label_dic)
            with open(combined_folder_path + 'ground_truth_labels/' + label_name, "w") as outfile:
                outfile.write(json_object)
            pred_label_path = pred_labels_path + label_name
            pred_label_dic = convert_string_to_tuples(pred_label_path, isAddOne, True)
            json_object = json.dumps(pred_label_dic)
            with open(combined_folder_path + 'predicted_labels/' + label_name, "w") as outfile:
                outfile.write(json_object)         

def display_results(all_iou, all_iou_count, all_dice, all_dice_count): 
    """
    Display the results of Intersection over Union (IoU) and Dice coefficient calculations for different classes.

    This function takes the results of IoU and Dice coefficient calculations for each class, along with the corresponding counts, and displays them in a formatted manner.

    Parameters:
    all_iou (list): A list containing the total IoU scores for each class.
    all_iou_count (list): A list containing the total counts for each class used in IoU calculations.
    all_dice (list): A list containing the total Dice coefficients for each class.
    all_dice_count (list): A list containing the total counts for each class used in Dice coefficient calculations.

    Returns:
    None
    """
    miou = 0
    label = ["APPAREL", "HAND\t", "PRODUCT"]
    for i in range(len(all_iou)):
        print("IoU of Class:", i, " Label: ", label[i], ": ", 0 if all_iou_count[i] == 0 else round(all_iou[i]/all_iou_count[i], 3), " \t Total-Count: ", all_iou_count[i])
    print("Mean IoU of all Classes : ", 0 if sum(all_iou_count) == 0 else round(sum(all_iou)/ sum(all_iou_count), 3))
    print("\n")
    for i in range(len(all_dice)):
        print("Dice of Class:", i, " Label: ", label[i], ": ", 0 if all_dice_count[i] == 0 else round(all_dice[i]/all_dice_count[i], 3), " \t Total-Count: ", all_dice_count[i])
    print("Mean Dice of all Classes : ", 0 if sum(all_dice_count) == 0 else round(sum(all_dice)/ sum(all_dice_count), 3))

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

def plot_results(iou_ans, task):
    """
    Plot the results of IoU (Intersection over Union) and Dice calculations.

    Parameters:
    iou_ans (list of 2D arrays): List containing 2D arrays of values for each thresholds.
    task (str): The type of task for which results are plotted ('IoU' , 'Dice').

    Returns:
    None: This function generates and saves plot images for each task.
    """
    x_axis = [round(value, 2) for value in np.arange(0, 1, 0.05)]
    titles = ["APPAREL", "HAND", "PRODUCT", "Mean"]
    for ctr, idv in enumerate(iou_ans):
        idv = idv[::-1]
        plt.rcParams["figure.figsize"] = (10, 10)
        fig, ax = plt.subplots()
        im = ax.imshow(idv, interpolation='none')

        # Set x and y ticks with labels
        ax.set_xticks(np.arange(len(x_axis)), labels=x_axis)
        ax.set_yticks(np.arange(len(x_axis)), labels=x_axis[::-1])

        # Add text annotations to the plot
        for i in range(0, 20):
            for j in range(0, 20):
                text = ax.text(i, j, round(idv[j][i], 1), ha="center", va="center", color="w")

        # Set title and labels
        plt.title(titles[ctr]+ " "+ task+ " Score")
        plt.xlabel("Confidence Threshold")
        plt.ylabel(task+" Threshold")
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(task + ' Score')
        plt.savefig(task+'_'+titles[ctr]+'_'+'plot.png') 

        print("For: ", titles[ctr])
        min_row, max_row, min_col, max_col = top_6_elements(idv)
        print("Range of IoU (rows): ", min_row," : ", max_row )
        print("Range of Confidence (cols): ", min_col, " : ",max_col)

if __name__ == "__main__": 
    # # Use storeDataFrame only once to get store ground truths and prediction in combined folder
    # storeDataFrame()

    # Define paths and parameters
    labels_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/combined_valid_segment/ground_truth_labels/"
    predictions_folder_path = "/home/pratham/walmart_poc/yolov7/seg/data/combined_valid_segment/predicted_labels/"
    iou_threshold = 0.5
    confidence_threshold = 0.5
    dice_threshold = 0.5 

    all_iou = [0, 0, 0]
    all_iou_count = [0, 0, 0]
    all_dice =  [0, 0, 0]
    all_dice_count =  [0, 0, 0]
    
    iou_ans = np.zeros((4, 20, 20)) # apparel, hand, product, mean
    dice_ans = np.zeros((4, 20, 20))
    for i in tqdm(range(0, 20)):
        for j in range(0,20):
            iou_threshold = i * 0.05
            confidence_threshold = j * 0.05
            for label in range(0,3):
                for label_name in sorted(os.listdir(labels_folder_path)):
                    file_path_gt = labels_folder_path + label_name
                    file_path_pred = predictions_folder_path + label_name

                    # Load ground truth and predicted polygon coordinates
                    ground_truth_polygons_string = open(file_path_gt).read()
                    ground_truth_polygons = json.loads(ground_truth_polygons_string)
                    predicted_polygons_string = open(file_path_pred).read()
                    predicted_polygons = json.loads(predicted_polygons_string)

                    # Compute accuracy metrics
                    iou, iou_count = calculate_polygon_iou(ground_truth_polygons, predicted_polygons, label, iou_threshold, confidence_threshold)
                    dice, dice_count = calculate_polygon_dice(ground_truth_polygons, predicted_polygons, label, iou_threshold, confidence_threshold)
                    all_iou[label]+= iou
                    all_iou_count[label]+=iou_count
                    all_dice[label]+=dice
                    all_dice_count[label]+=dice_count
                iou_ans[label][i][j] = round(all_iou[label]/all_iou_count[label], 3)
                dice_ans[label][i][j] = round(all_dice[label]/all_dice_count[label], 3)
            iou_ans[3][i][j] = round(sum(all_iou)/ sum(all_iou_count), 3)
            dice_ans[3][i][j] = round(sum(all_dice)/ sum(all_dice_count), 3)
    plot_results(iou_ans, "IoU")
    plot_results(dice_ans, "Dice")
