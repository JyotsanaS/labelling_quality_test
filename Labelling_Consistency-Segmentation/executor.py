import mistake_score.constants as constants
from distribution.aws_client import Aws, S3Downloader
from mistake_score.mistake_score import MistakeScore
from tqdm import tqdm
import botocore
import time
import os
import torch
import boto3
import tempfile


class Executor:
    def __init__(self):
        self.generator = None

    def initialise(self, initArgs, role, auth):

        # device = "cpu"  # torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if initArgs["model_variant"] == "small":
            model_type = "vit_b"
            http_path = "https://raga-satsure.s3.us-east-2.amazonaws.com/labelling_quality/mistake_score/sam_vit_b_01ec64.pth"
            sam_parameters = {
                "points_per_side": 16,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.80,  # 0.92,
                "crop_n_layers": 0,
                "crop_n_points_downscale_factor": 0,
                "min_mask_region_area": 100,
            }
        elif initArgs["model_variant"] == "medium":
            model_type = "vit_b"
            http_path = "https://raga-satsure.s3.us-east-2.amazonaws.com/labelling_quality/mistake_score/sam_vit_b_01ec64.pth"
            sam_parameters = {
                "points_per_side": 16,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.80,  # 0.92,
                "crop_n_layers": 1,
                "crop_n_points_downscale_factor": 2,
                "min_mask_region_area": 100,
            }
        else:
            model_type = "vit_h"
            http_path = "https://raga-satsure.s3.us-east-2.amazonaws.com/labelling_quality/mistake_score/sam_vit_h_4b8939.pth"  # "s3://raga-engineering/satsure_lqa/sam_vit_h_4b8939.pth"
            sam_parameters = {
                "points_per_side": 32,
                "pred_iou_thresh": 0.80,
                "stability_score_thresh": 0.80,  # 0.92,
                "crop_n_layers": 1,
                "crop_n_points_downscale_factor": 2,
                "min_mask_region_area": 100,
            }

        checkpoint_path = self.download_checkpoint(http_path)
        # checkpoint_path = "/home/ubuntu/1.1Tdisk/Satsure_n_channel/SAM_model/sam_vit_h_4b8939.pth"  # upload checkpoint to aws and use url to downalod it in temp and use its path from local
        # shutil.rmtree(new_annotation_dir) tempfile

        self.aws_client = Aws(role, auth)
        self.s3_downloader = self.aws_client.get_s3_client()

        self.generator = MistakeScore(
            # self.s3_downloader,
            device=initArgs["device"],
            model_type=model_type,
            checkpoint_path=checkpoint_path,
            sam_parameters=sam_parameters,
            # **initArgs,
        )

    def run(self, data_frame, input_args, output_args, role=None, auth=None):
        if self.generator is None:
            raise RuntimeError("You must call 'initialise' before running this method.")

        # Apply function with tqdm for the progress bar
        tqdm.pandas(desc="Processing Images")  # Initialize tqdm for pandas apply
        image_path = data_frame[input_args[constants.IMG_PATHS]].progress_apply(
            lambda x: self.download_file(x)
        )
        anns_path = data_frame[input_args[constants.ANNS_PATH]].progress_apply(
            lambda x: self.download_file(x)
        )
        mistake_scores = []
        for img_path, ann_path in zip(image_path, anns_path):
            mistake_scores.append(
                self.generator.generate_mistake_score(img_path, ann_path)
            )
        data_frame[output_args[constants.MISTAKESCORE]] = mistake_scores
        # data_frame[output_args[constants.MISTAKESCORE]] = data_frame[
        #     input_args[constants.IMG_PATHS]
        # ].progress_apply(
        #     lambda x: self.generator.generate_mistake_score(self.download_file(x))
        # )
        return data_frame

    def replace_url(self, http_path):
        s3_path = http_path.replace(
            "https://raga-satsure.s3.us-east-2.amazonaws.com", "s3://raga-satsure"
        )
        return s3_path

    def download_checkpoint(self, http_path):
        s3_path = self.replace_url(http_path)
        bucket, key = s3_path.replace("s3://", "").split("/", 1)

        s3 = boto3.client("s3")

        _, temp_file_path = tempfile.mkstemp()

        with open(temp_file_path, "wb") as f:
            s3.download_fileobj(bucket, key, f)
        return temp_file_path

    def download_file(self, x, max_retries=5, retry_delay=1):
        retries = 0

        while retries < max_retries:
            try:
                return self.s3_downloader.download_pass(x)
            except botocore.exceptions.ClientError as e:
                self.s3_downloader = self.aws_client.get_s3_client()
                error_code = e.response["Error"]["Code"]
                if error_code == "NoSuchKey":
                    print(f"File not found: {x}")
                    break  # Do not retry if the file does not exist
                print(f"S3 Download failed with error code {error_code}: {e}")
            except Exception as e:
                self.s3_downloader = self.aws_client.get_s3_client()
                print(f"Download failed: {e}")

            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        # If all retries fail, raise the last exception
        raise Exception("Max retries reached. Download failed.")

    def echo(self):
        print("Wheel File working")
