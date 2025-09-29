import pandas as pd
from tqdm import tqdm
from PIL import Image, ImageOps
from .transforms import get_bbox_coordinates
import os
import glob

class SaveROI:
    def __init__(self, annotation_csv, image_path, save_path, method="resize"):
        """
        
        Initialize the class with paths and method for cropping.

        Parameters:
            annotation_csv (str): Path to the CSV file containing annotations.
            image_path (str): Path to the folder containing images.
            save_path (str): Path where cropped images will be saved.
            method (str): Cropping method, defaults to "resize".
        """
        self.anno_df = pd.read_csv(annotation_csv)
        self.anno_df['annotation_issue'] = self.anno_df['annotation_issue'].apply(lambda x: eval(x))
        self.image_path = glob.glob(image_path)
        self.save_path = save_path
        self.method = method

    def return_resize_roi(self, img, xmin, ymin, xmax, ymax):
        """
        Resize the region of interest (ROI) to a fixed size of 224x224.

        Parameters:
            img (Image): The source image.
            xmin, ymin, xmax, ymax (int): Coordinates for the ROI.

        Returns:
            Image: Cropped and resized image.
        """
        new_width, new_height = 224, 224
        crop_img = img.crop((xmin, ymin, xmax, ymax)).resize((new_width, new_height))
        return crop_img
    
    def return_aspect_ratio_crop(self, img, xmin, ymin, xmax, ymax):
        """
        Maintain the aspect ratio of the ROI and pad to create a square image.

        Parameters:
            img (Image): The source image.
            xmin, ymin, xmax, ymax (int): Coordinates for the ROI.

        Returns:
            Image: Cropped image with maintained aspect ratio.
        """
        roi_width = xmax - xmin
        roi_height = ymax - ymin
        scale = 224 / max(roi_width, roi_height)
        new_width = int(roi_width * scale)
        new_height = int(roi_height * scale)
        cropped_image = img.crop((xmin, ymin, xmax, ymax)).resize((new_width, new_height))
        if new_width != new_height:
            max_size = max(new_width, new_height)
            padded_image = Image.new('RGB', (max_size, max_size), (0, 0, 0))
            upper_x = (max_size - new_width) // 2
            upper_y = (max_size - new_height) // 2
            padded_image.paste(cropped_image, (upper_x, upper_y))
            return padded_image
        return cropped_image
    
    def return_square_crop(self, img, xmin, ymin, xmax, ymax):
        """
        Crop the image into a fixed square of size 224x224 centered around the ROI.

        Parameters:
            img (Image): The source image.
            xmin, ymin, xmax, ymax (int): Coordinates for the ROI.

        Returns:
            Image: Cropped square image.
        """
        center_x, center_y = (xmin + xmax) // 2, (ymin + ymax) // 2
        xmin = max(0, center_x - 112)
        xmax = xmin + 224
        ymin = max(0, center_y - 112)
        ymax = ymin + 224
        if xmax > img.width:
            xmax = img.width
            xmin = xmax - 224
        if ymax > img.height:
            ymax = img.height
            ymin = ymax - 224
        cropped_image = img.crop((xmin, ymin, xmax, ymax))
        return cropped_image

    def return_pad_crop(self, img, xmin, ymin, xmax, ymax):
        """
        Crop the image and pad it to a square of 224x224.

        Parameters:
            img (Image): The source image.
            xmin, ymin, xmax, ymax (int): Coordinates for the ROI.

        Returns:
            Image: Padded image.
        """
        cropped_image = img.crop((xmin, ymin, xmax, ymax))
        roi_width = int(xmax - xmin)
        roi_height = int(ymax - ymin)
        if roi_width < 224 or roi_height < 224:
            cropped_image = cropped_image.resize((max(roi_width, 224), max(roi_height, 224)))
        padded_image = ImageOps.pad(cropped_image, (224, 224), centering=(0.5, 0.5))
        return padded_image
    
    def save_roi(self):
        """
        Process each image and save the cropped regions according to the specified method.
        """
        for image_path in tqdm(self.image_path):
            img = Image.open(image_path)
            width, height = img.size
            filename = os.path.basename(image_path)
            image_id = filename.replace('.jpg', '')
            df = self.anno_df[self.anno_df['filepath'] == image_path]

            for i, row in df.iterrows():
                for annotation in row['annotation_issue']:
                    xmin, ymin, xmax, ymax = get_bbox_coordinates(annotation)
                    xmin, ymin, xmax, ymax = xmin * width, ymin * height, xmax * width, ymax * height

                    if self.method == "resize":
                        roi_img = self.return_resize_roi(img, xmin, ymin, xmax, ymax)
                    elif self.method == "aspect_ratio":
                        roi_img = self.return_aspect_ratio_crop(img, xmin, ymin, xmax, ymax)
                    elif self.method == "square":
                        roi_img = self.return_square_crop(img, xmin, ymin, xmax, ymax)
                    elif self.method == "padding":
                        roi_img = self.return_pad_crop(img, xmin, ymin, xmax, ymax)
                    else:
                        raise ValueError("Unknown method")

                    name = annotation['class_name']
                    crop_no = annotation['class_id']
                    imagename = f'{image_id}-{crop_no}.jpg'
                    category_path = os.path.join(self.save_path, name)

                    if not os.path.exists(category_path):
                        os.makedirs(category_path)

                    roi_img.save(os.path.join(category_path, imagename))

if __name__ == "__main__":
    annotation_csv = "/home/ubuntu/LQC/dataset/data_files/pascal_dataset.csv"
    image_path = '/home/ubuntu/LQC/dataset/pascal/VOC2012/JPEGImages/*.jpg'
    save_path = '/home/ubuntu/LQC/dataset/pascal/roi/padding'
    save_cropped = SaveROI(annotation_csv, image_path, save_path, "padding")
    save_cropped.save_roi()
