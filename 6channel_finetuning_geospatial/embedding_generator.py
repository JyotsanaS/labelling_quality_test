import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from tifffile import imread
import numpy as np
import torch.nn.functional as F

class EmbeddingGenerator:
    def __init__(
        self, model, folder_path, json_filename, batch_size=32, device=None
    ):
        self.folder_path = folder_path
        self.batch_size = batch_size

        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        print(self.device)

#         self.transforms = transforms.Compose(
#             [
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
#             ]
#         )

        self.model = model.dinov2

        self.data = self._get_embeddings_for_folder()
        with open(json_filename, "w") as json_file:
            json.dump(self.data, json_file, indent=4)


    def preprocess(self, image_path):
        image = imread(image_path)
        image = image.astype(float)
        tensor_image = torch.tensor(image, dtype=torch.float32)
        # Define the new dimensions
        new_dimensions = (64, 64)
        
        # Resize the 6-channel image using torch.nn.functional.interpolate
        resized_tensor = F.interpolate(
            torch.unsqueeze(tensor_image.permute(2, 0, 1), 0),  # Add batch and channel dimensions
            size=new_dimensions,
            mode='bilinear',
            align_corners=False
        )

        # Remove batch dimension and transpose back to original shape
        resized_image = resized_tensor.squeeze().permute(1, 2, 0).numpy()
        image = np.transpose(resized_image, (2, 1, 0))

        image = (image - image.min()) / (image.max() - image.min())
        image = torch.from_numpy(np.flip(image,axis=0).copy())

        return image

    def _run_model(self, image_tensor):
        self.model = self.model.to(self.device)
        with torch.no_grad():
            features = self.model(image_tensor.to(self.device).float())['pooler_output']
            features = torch.nn.functional.normalize(features, dim=1)

        return features.detach().cpu().numpy()

    def _get_embeddings_for_folder(self):
#         folders = [
#             item
#             for item in os.listdir(self.folder_path)
#             if os.path.isdir(os.path.join(self.folder_path, item))
#         ]
        folders = ['images']

        data = []
        for folder in folders:
            filenames = [
                f
                for f in os.listdir(os.path.join(self.folder_path, folder))
                if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".tif")
            ]

            num_files = len(filenames)
            for i in tqdm(
                range(0, num_files, self.batch_size), desc=f"{folder}({num_files})"
            ):
                batch_files = filenames[i : i + self.batch_size]
                actual_batch_size = len(batch_files)

                batch_tensors = [
                    self.preprocess(os.path.join(self.folder_path, folder, f))
                    for f in batch_files
                ]
                
                batch_tensor = torch.stack(batch_tensors, 0)
                embeddings = self._run_model(batch_tensor)

                for j in range(actual_batch_size):
                    t = {
                        "id": batch_files[j].split(".")[0],
                        "filepath": os.path.join(
                            self.folder_path, folder, batch_files[j]
                        ),
                        "label": folder,
                        "pred": folder,
                        "embedding": embeddings[j].tolist(),
                    }

                    data.append(t)
        return data