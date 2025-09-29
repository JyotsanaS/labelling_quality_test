from TestingPackages.ood_mahalanobis import OOD
from plot_outliers import plot_outlier_images
import pandas as pd
import plotly.express as px
import cv2
import numpy as np
import matplotlib.pyplot as plt
import re 
import os
from PIL import Image
from matplotlib import patches

segment_colour = {
    'product': 'green',
    'hand': 'red',
    'apparel': 'blue'
}

def get_original_filename(segmented_filename):
    # Use regular expression to extract the original filename
    match = re.search(r'(.+)_segmented_\d+\.jpg', segmented_filename)
    if match:
        return match.group(1)
    else:
        return None

def read_text_file(file_path):
    segmentations = []

    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split()
            segmentation = list(map(float, values[1:]))
            
            segmentations.append(segmentation)

    return segmentations

def draw_segmentations(image, segmentation, label, transparency=1):

    # Rescale the normalized coordinates to match the image size
    w, h = image.size
    points = np.array(segmentation).reshape(-1, 2) * np.array([w, h])

    # Create a polygon patch
    polygon = patches.Polygon(points, closed=True, edgecolor='k', linewidth=2, facecolor=segment_colour[label],
                              alpha=transparency)
    
    return polygon

def perform_outlier(filepath):
    df = pd.read_json(filepath)

    # List of unique labels in the 'label' column
    unique_labels = df['label'].unique()

    for label in unique_labels:
        # Subset the dataframe for the current label
        df_label = df[df['label'] == label]

        # Perform outlier detection for the current label
        od = OOD(df_label, {'threshold': 5})
        df_ood_label = df_label.copy()

        ood_score = []
        for i, row in df_label.iterrows():
            _, score = od.check_ood(row['embedding'])
            ood_score.append(score)

        df_ood_label['score'] = ood_score

        # Plot distribution of scores using Plotly
        fig = px.histogram(df_ood_label, x='score', nbins=30,
                           title=f'Distribution of Mahalanobis Distance Scores for Label: {label}',
                           labels={'score': 'Score', 'count': 'Frequency'})
        fig.show()

        max_dis = float(input(f"Enter the max Mahalanobis distance(d) for Label {label}: "))
        result_df = df_ood_label.sort_values(by='score', ascending=False)
        result_df = result_df[result_df['score'] > max_dis]

        # Optionally, you can still plot the outlier images
        for i, row in result_df.iterrows():
            image_path = row['filepath']
            label = image_path.split('/')[-2]
            folder = image_path.split('/')[-5]
            set_name = image_path.split('/')[-4]

            label_filename = get_original_filename(image_path.split('/')[-1]) 
            label_path = os.path.join('./dataset_whole/training_valid_dataset', folder, set_name,'labels', 
                                      label_filename + '.txt')  # Replace with the actual path to your label files
            whole_image_path = os.path.join('./dataset_whole/training_valid_dataset', folder, set_name
                                            ,'images', label_filename + '.jpg')

            segmented_no = int(image_path.split('_')[-1].replace('.jpg', ''))  # Extract segmented number from the file path
            segmented_image = Image.open(image_path)
            original_image = Image.open(whole_image_path)

            segment = read_text_file(label_path)[segmented_no-1]

            # Display the original image, highlighted image, and cropped image
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            axs[0].imshow(segmented_image)
            axs[0].set_title(f'segmented image label: {label}')

            polygon = draw_segmentations(original_image, segment, label)
            axs[1].imshow(original_image)
            axs[1].add_patch(polygon)


            plt.show()