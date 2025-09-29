import numpy as np
import matplotlib.pyplot as plt
import json
from umap import UMAP
import matplotlib.image as mpimg
import plotly.express as px
import seaborn as sns
import pandas as pd


class OOD:
    def __init__(self, json_file, flag=True):
        # flag = false, then json_file = (embeddings, filepaths)
        self.json_file = json_file
        self.flag = flag
        self.embeddings, self.filepaths = self._read_json()
        print("No of Images: ", len(self.filepaths))

        # Calculate the mean and covariance of the train embedding.
        regularization_factor = 1e-6
        self.mean = np.mean(np.array(self.embeddings), axis=0)
        self.cov = np.cov(np.array(self.embeddings).T)
        self.cov = self.cov + np.eye(self.cov.shape[0]) * regularization_factor
        self.inv_cov = np.linalg.pinv(self.cov)

        self.scores = self.calculate_ood_score(self.embeddings)

    def _read_json(self):
        if self.flag:
            with open(self.json_file, "r") as f:
                data = json.load(f)
            embeddings = [d["embedding"] for d in data]
            filepaths = [d["filepath"] for d in data]
        else:
            embeddings, filepaths = self.json_file
        return embeddings, filepaths

    def calculate_ood_score(self, embeddings):
        # embeddings = np.array(self.embeddings)
        mean_diff = embeddings - self.mean
        scores = np.sqrt(np.sum(mean_diff @ self.inv_cov * mean_diff, axis=1))
        scores = scores.tolist()
        # print("ood scores claculated.\n")
        return scores

    def plot_score_histogram(self, percentile=99.9):
        THRESHOLD = np.percentile(np.array(self.scores), percentile)
        print(f"THRESHOLD: {THRESHOLD}")

        plt.hist(self.scores, bins=130, edgecolor="black")
        plt.xlabel("scores")
        plt.ylabel("frequency")
        # plt.title("scores histogram")

        # Draw a vertical red line at the threshold
        plt.axvline(x=THRESHOLD, color="red", linestyle="dashed", linewidth=2)
        plt.text(
            THRESHOLD,
            plt.gca().get_ylim()[1],
            f"Percentile: {percentile}\nThreshold value: {THRESHOLD:.2f}",
            color="red",
            ha="right",
        )
        plt.show()
        return THRESHOLD

    def plot_ood_drift(self, score2):
        score1 = self.scores
        percentile = 99.9
        THRESHOLD = np.percentile(np.array(score1), percentile)

        # Draw a vertical red line at the threshold
        plt.axvline(x=THRESHOLD, color="blue", linestyle="dashed", linewidth=2)

        plt.text(
            0.5,
            1.01,
            f"percentile: {percentile}\nthreshold: {THRESHOLD:.2f}",
            color="blue",
            ha="center",
            transform=plt.gca().transAxes,
        )

        plt.hist(
            score1,
            bins=130,
            edgecolor="black",
            color="black",
            alpha=0.7,
            label="reference",
        )
        plt.hist(
            score2, bins=130, edgecolor="red", color="red", alpha=0.7, label="field"
        )
        plt.xlabel("ood scores")
        plt.ylabel("Frequency")
        # plt.title('product class')
        plt.legend()
        plt.show()

        sns.kdeplot(score1, bw_adjust=0.5, label="reference", color="black")
        sns.kdeplot(score2, bw_adjust=0.5, label="field", color="red")
        plt.xlabel("ood score")
        plt.ylabel("probability density")
        plt.legend()
        plt.show()

    def plot_umap(self, embeddings=None, labels=None):
        if not embeddings:
            embeddings = self.embeddings

        umap = UMAP(n_components=2, n_neighbors=15, random_state=42)
        umap_data_2d = umap.fit_transform(embeddings)

        fig = px.scatter(
            umap_data_2d, x=0, y=1, color=labels, labels={"color": "ood detection"}
        )
        fig.show()

    def get_labels(self, THRESHOLD=100):
        return [True if s > THRESHOLD else False for s in self.scores]

    def plot_top_ood(self, n_rows=4, n_cols=4, data=None, reverse=True):
        # if data is given then plot from that data else plot from reference data
        results = []
        if data:
            scores = data[0]
            filepaths = data[1]
        else:
            scores = self.scores
            filepaths = self.filepaths

        for score, file in zip(scores, filepaths):
            results.append([score, file])
        results.sort(reverse=reverse)

        n = min(n_rows * n_cols, len(results))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
        for i, ax in enumerate(axes.flatten()):
            if i < n:
                img_path = results[i][1]
                score = results[i][0]
                img = mpimg.imread(img_path)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(f"score: {float(score):.5f}")
        plt.show()

    def save_score_csv(self, filepaths, scores, save_path):
        # Create a DataFrame
        data = {"filepath": filepaths, "score": scores}
        df = pd.DataFrame(data)

        # Sort the DataFrame by the 'score' column
        df = df.sort_values(by="score", ascending=False)

        # Save to CSV
        df.to_csv(save_path, index=False)
