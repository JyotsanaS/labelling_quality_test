import argparse
import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.brain as fob


def parse_arguments():
    parser = argparse.ArgumentParser(description="Process command-line arguments")
    parser.add_argument("--path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument(
        "--port", type=int, required=True, help="Port for launching the app"
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    path = args.path
    port = args.port

    name = path.rsplit("/")[-1].replace(".csv", "").split("-")[0] + path.split("/")[-1].split("_")[-2]

    df = pd.read_json(path)

    annotations_list = dict(df[["filepath", "label"]].values)
    embeddings = np.array(df["embedding"].values.tolist())

    # Create samples for your data
    samples_base = []
    for filepath in df["filepath"].values.tolist():
        sample = fo.Sample(filepath=filepath)

        # Store classification in a field name of your choice
        label = str(annotations_list[filepath])
        sample["ground_truth"] = fo.Classification(label=label)

        samples_base.append(sample)

    # Create dataset
    dataset = fo.Dataset(name)
    dataset.add_samples(samples_base)

    # Compute similarity with Qdrant backend
    fob.compute_similarity(
        dataset,
        embeddings=embeddings,
        brain_key="qdrant_example",
        url="http://localhost:6333",
        backend="qdrant",
    )

    # Compute 2D representation
    fob.compute_visualization(
        dataset,
        embeddings=embeddings,
        num_dims=2,
        method="umap",
        brain_key=name,
        verbose=True,
        seed=51,
    )

    session = fo.launch_app(dataset, port=port)
    session.wait()


if __name__ == "__main__":
    main()
