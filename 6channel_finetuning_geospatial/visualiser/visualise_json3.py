import os

print(os.getcwd())
import sys

sys.path.append(os.getcwd())


import pandas as pd
from visualiser.app import App


def visualise_embeddings(parent_path, port=8051):
    dataset_dfs = {}
    root_paths = {}
    # for rl in range(4):
    #     for fl in range(7):
    name1 = f"embeddings_satsure_tiles_finetune-model-99_small.json"
    json_path1 = f"{parent_path}/{name1}"
    df1 = pd.read_json(json_path1)
    dataset_dfs[f"embeddings_satsure"] = df1
    root_paths["embeddings_satsure"] = ""

    # for two dataset: dropdown
    # name2 = "grass_barren_final_results_60_70_15.json"
    # json_path2 = f"{parent_path}/{name2}"
    # df2 = pd.read_json(json_path2)
    # dataset_dfs[f"grass_barren_final_results_60_70_15"] = df2
    # root_paths["grass_barren_final_results_60_70_15"] = ""

    # print(dataset_dfs.keys())
    # return

    app = App(
        dataframes_dict=dataset_dfs,
        root_paths=root_paths,
    )

    # Run the app
    app.run(host="0.0.0.0", debug=True, use_reloader=False, port=port)


if __name__ == "__main__":
    # OpenAI CLIP
    visualise_embeddings(
        "/home/ubuntu/home/ubuntu/6channel_training/embeddings_dinov2",
        port=8063,
    )
