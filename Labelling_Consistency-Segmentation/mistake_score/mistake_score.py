import os
import cv2
import pdb
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
import tifffile as tif
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import pdb
import boto3

# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from matplotlib.colors import ListedColormap


class MistakeScore:
    def __init__(
        self,
        device,
        model_type,
        checkpoint_path,
        sam_parameters,
    ):
        self.device = device
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam_parameters = sam_parameters
        # pdb.set_trace()
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path).to(
            device=self.device
        )
        self.mask_generator_2 = SamAutomaticMaskGenerator(model=sam, **sam_parameters)

    def get_imgs_anns_list(self, images_folder_path, anns_folder_path):
        if os.path.isdir(images_folder_path) and os.path.isdir(anns_folder_path):
            images_path_list = [
                file
                for ext in ["tif", "jpg", "png"]
                for file in glob(f"{images_folder_path}/*.{ext}")
            ]
            # images_path_list = glob(f"{images_folder_path}/*.tif") + glob(f"{images_folder_path}/*.jpg") + glob(f"{images_folder_path}/*.png")
            anns_path_list = glob(f"{anns_folder_path}/*.tif")
        elif os.path.isfile(images_folder_path) and os.path.isfile(anns_folder_path):
            images_path_list = [images_folder_path]
            anns_path_list = [anns_folder_path]
        return sorted(images_path_list), sorted(anns_path_list)

    def read_image(self, image_path):
        extension = os.path.splitext(image_path)[-1][1:]
        try:
            if extension == "tif":
                image = tif.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif extension in ["png", "jpg", "jpeg"]:
                image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image.shape[-1] > 3:
                image = image[..., :3]
            # else:
            #     print(f"Number of channel not supported: {image.shape[-1]}")
        except Exception as e:
            print(f"Unsupported image extension: {extension}")
        return image

    def get_matching_imgs_anns(self, images_folder_path, anns_folder_path):
        images_path_list, anns_path_list = self.get_imgs_anns_list(
            images_folder_path, anns_folder_path
        )
        if len(images_path_list) == 1 and len(anns_path_list) == 1:
            return images_path_list, anns_path_list
        else:
            image_paths1 = [
                os.path.splitext(os.path.basename(item))[0] for item in images_path_list
            ]
            anns_paths1 = [
                os.path.splitext(os.path.basename(item))[0] for item in anns_path_list
            ]
            common_elements = set(image_paths1) & set(anns_paths1)
            filtered_image_path_list = [
                img_path
                for img_path in images_path_list
                if os.path.splitext(os.path.basename(img_path))[0] in common_elements
            ]
            filtered_anns_path_list = [
                ann_path
                for ann_path in anns_path_list
                if os.path.splitext(os.path.basename(ann_path))[0] in common_elements
            ]
            return sorted(filtered_image_path_list), sorted(filtered_anns_path_list)

    def get_new_annotation(self, annotation, masks):
        # create new annotation having same shape as that of annotation but filled with -1 value
        new_annotation = np.ones_like(annotation) * -1
        for i, mask in enumerate(masks):
            boolean_mask = mask["segmentation"]
            # pdb.set_trace()
            unique_values, counts = np.unique(
                annotation[boolean_mask], return_counts=True
            )

            max_count_index = np.argmax(counts)

            max_value = unique_values[max_count_index]
            max_count = counts[max_count_index]

            new_annotation[boolean_mask] = max_value
        return new_annotation

    def compare_annotations(self, np1, np2, no_data_classes=[]):
        #     unique_labels = np.unique(np1)
        unique_labels = np.union1d(np.unique(np1), np.unique(np2))
        result = {}

        for label in unique_labels:
            label_mask = np1 == label
            total_pixels = np.sum(label_mask)
            differing_pixels = np.sum(label_mask & (np1 != np2)) | np.sum(
                np2 == label & (np1 != np2)
            )
            percent_difference = (
                (differing_pixels / total_pixels) if total_pixels != 0 else 0
            )

            result[label] = {
                "diff": percent_difference,
                "pixel_area": total_pixels / (np1.shape[0] * np1.shape[1]),
            }

        if -1 in result.keys():
            del result[-1]
        for label in no_data_classes:
            del result[label]
        return result

    def update_mistake_score(self, ms):
        mistake_scores = {}
        pixel_areas = {}
        for key in ms.keys():
            mistake_scores[key] = ms[key]["diff"]
            pixel_areas[key] = ms[key]["pixel_area"]
        return {"mistake_scores": mistake_scores, "pixel_areas": pixel_areas}

    # def update_mistake_score_dict(self, mistake_score):
    #     labelmap = self.labelmap
    #     total_labelmap = len(list(labelmap.keys()))
    #     ms = {}
    #     pixel = {}
    #     for i in range(total_labelmap):
    #         # i = str(i)
    #         val = mistake_score.get(i, -1)
    #         if val != -1:
    #             ms[i] = val["diff"]
    #             pixel[i] = val["pixel_area"]
    #         else:
    #             ms[i] = -1
    #             pixel[i] = -1
    #     return {"mistake_scores": ms, "pixel_areas": pixel}

    def get_mistake_scores(self, images_folder_path, anns_folder_path):
        images_path_list, anns_path_list = self.get_matching_imgs_anns(
            images_folder_path, anns_folder_path
        )
        data = []
        for (
            idx,
            image_path,
        ) in enumerate(images_path_list):

            image = self.read_image(image_path)
            annotation_path = anns_path_list[idx]
            annotation = tif.imread(annotation_path)
            # pdb.set_trace()
            masks_2 = self.mask_generator_2.generate(image)

            new_annotation = self.get_new_annotation(annotation, masks=masks_2)
            annotation_comparison = self.compare_annotations(annotation, new_annotation)
            data.append(
                {
                    "id": idx,
                    "image_path": image_path,
                    "annotation_path": annotation_path,
                    "annotation_comparison": annotation_comparison,
                }
            )
        return data

    def get_update_mistake_score_df(self, data):
        df = pd.DataFrame(data)
        updated_mistake_score = []
        for _, row in df.iterrows():
            mistake_score = row["annotation_comparison"]
            updated_mistake_score.append(self.update_mistake_score(mistake_score))
        df["mistake_scores"] = updated_mistake_score
        df.drop(["annotation_comparison"], axis=1, inplace=True)
        return df

    def generate_mistake_score(self, images_folder_path, anns_folder_path):
        data = self.get_mistake_scores(images_folder_path, anns_folder_path)
        df = self.get_update_mistake_score_df(data)
        return df["mistake_scores"].iloc[0]


# if __name__ == "__main__":
#     # Example for one image and one ann path path
#     df = pd.read_csv(
#         "/home/ubuntu/1.1Tdisk/Satsure_n_channel/datasets/lqa/df_final.csv"
#     )
#     s3_img_path = df.img_path.iloc[0]
#     s3_ann_path = df.label_path.iloc[0]

#     def download_s3_file_to_folder(s3_path, output_folder):
#         bucket, key = s3_path.replace("s3://", "").split("/", 1)

#         s3 = boto3.client("s3")

#         os.makedirs(output_folder, exist_ok=True)

#         output_file_path = os.path.join(
#             output_folder, s3_path.split("/")[-2] + os.path.basename(key)
#         )

#         s3.download_file(bucket, key, output_file_path)

#         return output_file_path

#     output_folder = (
#         "/home/ubuntu/1.1Tdisk/Satsure_n_channel/SAM_model/SAM_mistake_score_wheel"
#     )
#     print("s3_img_path", s3_img_path)
#     print("s3_ann_path", s3_ann_path)
#     images_folder_path = download_s3_file_to_folder(s3_img_path, output_folder)
#     anns_folder_path = download_s3_file_to_folder(s3_ann_path, output_folder)
#     print("images_folder_path", images_folder_path)
#     print("anns_folder_path", anns_folder_path)

#     # Example for images folder path and anns folder path
#     # images_folder_path = "/home/ubuntu/1.1Tdisk/Satsure_n_channel/datasets/lqa/images"
#     # anns_folder_path = (
#     #     "/home/ubuntu/1.1Tdisk/Satsure_n_channel/datasets/lqa/LULC_9_classes"
#     # )
#     checkpoint_path = (
#         "/home/ubuntu/1.1Tdisk/Satsure_n_channel/SAM_model/sam_vit_h_4b8939.pth"
#     )
#     device = "cpu"
#     model_type = "vit_h"
#     sam_parameters = {
#         "points_per_side": 16,
#         "pred_iou_thresh": 0.80,
#         "stability_score_thresh": 0.80,  # 0.92,
#         "crop_n_layers": 1,
#         "crop_n_points_downscale_factor": 2,
#         "min_mask_region_area": 100,
#     }
#     obj = MistakeScore(
#         device,
#         model_type,
#         checkpoint_path,
#         sam_parameters=sam_parameters,
#     )
#     print(obj.generate_mistake_score(images_folder_path, anns_folder_path))

# labelmap = {
#     "0": "no data",
#     "1": "water",
#     "2": "aquaculture",
#     "3": "crops(farmlands,horticulture) ",
#     "4": "bare ground",
#     "5": "built",
#     "6": "forest",
#     "7": "plantations-(large orchids, vineyards)",
#     "8": "trees",
# }
