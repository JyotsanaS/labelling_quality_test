import executor
import pandas as pd

obj = executor.Executor()
obj.initialise({"labelmap": {"0": "1", "1": "2"}}, None, None)
df = pd.read_csv(
    "/home/ubuntu/1.1Tdisk/walmart_poc/data_quality_checks/data_upload_csvs/roi_level_drift_with_masked_embeddings/drift_roi_train_masked_final.csv"
).head(5)
input_args = "image_path"
input_args = {"image_path": "masked_img_paths", "ann_path": "one_object_anns_tif"}
output_args = {"mistake_scores": "mistake_scores"}
obj.run(df, input_args, output_args)
# print(abc)
