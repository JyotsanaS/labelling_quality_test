import fiftyone as fo
import glob
import fiftyone.brain as fob
import pandas as pd
import numpy as np
from fiftyone import ViewField as F

images_patt = "/home/ubuntu/home/ubuntu/6channel_training/dataset_tiles/satsure_small_rgb_jpg/images/*"

df_base = pd.read_json('/home/ubuntu/home/ubuntu/6channel_training/embeddings_dinov2/embeddings_satsure_tiles_base-model_small_jpg.json')
df_finetuned = pd.read_json('/home/ubuntu/home/ubuntu/6channel_training/embeddings_dinov2/embeddings_satsure_tiles_finetune-model-99_small_jpg.json')

annotations_base = dict(df_base[['filepath', 'label']].values)
embeddings_base = np.array(df_base['embedding'].values.tolist())

# annotations_finetuned = dict(df_finetuned[['filepath', 'label']].values)
embeddings_finetuned = np.array(df_finetuned['embedding'].values.tolist())

# embeddings_finetuned_2 = np.array(df_finetuned_2['embedding'].values.tolist())

# Create samples for your data
samples_base = []
for filepath in annotations_base.keys():
    sample = fo.Sample(filepath=filepath)

    # Store classification in a field name of your choice
    label = str(annotations_base[filepath]) + '_base'
    sample["ground_truth"] = fo.Classification(label=label)

    samples_base.append(sample)
    
# Create dataset
dataset_base = fo.Dataset("satsure_6channel_small")
dataset_base.add_samples(samples_base)
    
# Compute 2D representation
results_base = fob.compute_visualization(
    dataset_base,
    embeddings=embeddings_base,
    num_dims=2,
    method="umap",
    brain_key="satsure_6channel_small_base",
    verbose=True,
    seed=51,
)

results_embeddings = fob.compute_visualization(
    dataset_base,
    embeddings=embeddings_finetuned,
    num_dims=2,
    method="umap",
    brain_key="satsure_6channel_small_finetuned_99",
    verbose=True,
    seed=51,
)

# session = fo.launch_app(dataset)

if __name__ == "__main__":
    # Ensures that the App processes are safely launched on Windows
    session = fo.launch_app(dataset_base, port=5151)
#     session = fo.launch_app(dataset_finetuned, port=5152)
    session.wait()