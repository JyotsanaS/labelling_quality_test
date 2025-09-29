import json
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from finch import FINCH


class FinchLQC:
    def __init__(self, filepath, algo_type="multiple_centroid", threshold=0.05):
        """
        Labelling quality checks for image classification and object detection. Supports single and multiple classes.

        Parameters:
            json_path (str): The path to the JSON file containing the filepath, embedding and label for each image.
            algo_type (str, optional): single_centroid: use single centroid for each class.
                            multiple_centroid: use multiple centroids for each class.
            threshold (float, optional): # less than this %count will not be considered for centroid calculation within a class. Defaults to 0.05.
        """
        self.df = pd.read_csv(filepath)
        self.algo_type = algo_type
        self.threshold = threshold
        
        #Get required columns
        self.embeddings = np.stack(self.df['embedding'].apply(lambda x: np.array(eval(x)))).astype(np.float32)
        self.labels = self.df['annotation_issue'].apply(lambda x: eval(x)['class_name']).tolist()
        self.filepaths = self.df['filepath'].values
        
        #Encode labels
        self.encoded_labels, self.label_dict = self._encode_labels()
        self.no_of_class = max(self.encoded_labels) + 1

    def _encode_labels(self):
        label_set = set(self.labels)
        label_dict = {l: i for i, l in enumerate(label_set)}
        r_label_dict = {v: k for k, v in label_dict.items()}
        encoded_labels = [label_dict[l] for l in self.labels]
        return encoded_labels, r_label_dict

    def get_centroid_labels(self, embeddings):
        c, num_clust, req_c = FINCH(
            embeddings, verbose=True
        )
        n = len(num_clust)
        labels = c[:, n - 1]
        return labels, num_clust

    def _calculate_centroids_and_scores(self, threshold):
        clusters_labels = []
        class_weighted_avg = []

        dict = {k: [] for k in range(self.no_of_class)}
        for i, label in enumerate(self.encoded_labels):
            dict[label].append(self.embeddings[i])

        for class_label, c_embeddings in dict.items():
            if self.algo_type == "single_centroid":
                c_labels, num_clust = [0 for _ in range(len(c_embeddings))], [1]
            else:
                c_labels, num_clust = self.get_centroid_labels(np.array(c_embeddings))
            c_dict = {k: [] for k in range(num_clust[-1])}
            for i, c_label in enumerate(c_labels):
                c_dict[c_label].append(c_embeddings[i])

            total_size = len(c_embeddings)
            req_dict = {}
            for k, emb in c_dict.items():
                print(f"{len(emb)/total_size*100:.2f}%")
                if len(emb) / total_size > threshold:
                    req_dict[k] = emb
            print(
                "No of clusters in class {}: {}\n".format(
                    self.label_dict[class_label], len(req_dict)
                )
            )

            for k in range(len(req_dict)):
                clusters_labels.append(class_label)

            for _, emb in req_dict.items():
                class_embeddings = np.array(emb)
                distances = 1 - np.dot(class_embeddings, class_embeddings.T) / (
                    np.linalg.norm(class_embeddings, axis=1)[:, np.newaxis]
                    * np.linalg.norm(class_embeddings, axis=1)[np.newaxis, :]
                )
                weights = 1 / np.sum(distances, axis=1)
                weights /= np.sum(weights)
                weighted_avg = np.dot(weights, class_embeddings)
                class_weighted_avg.append(weighted_avg)

        scores = []
        class_weighted_avg = np.array(class_weighted_avg)
        for e in tqdm(self.embeddings):
            cosine_distances = 1 - np.dot(class_weighted_avg, e) / (
                np.linalg.norm(class_weighted_avg, axis=1) * np.linalg.norm(e)
            )
            if self.no_of_class != 1:
                normalized_distances = cosine_distances / np.sum(cosine_distances)
                rounded_normalized = np.round(normalized_distances, 10).tolist()
                scores.append(rounded_normalized)
            else:
                scores.append(cosine_distances)
        return scores, clusters_labels

    def save_dataframe(self, savepath):
        scores, clusters_labels = self._calculate_centroids_and_scores(self.threshold)
        print(clusters_labels)
        data = []
        for i in range(len(self.filepaths)):
            ids = self.filepaths[i].split("/")[-1].split(".")[0]
            filepath = self.filepaths[i]
            score = np.min(scores[i])
            pred = self.label_dict[clusters_labels[np.argmin(scores[i])]]
            label = self.labels[i]

            data.append(
                {
                    "ids": ids,
                    "filepath": filepath,
                    "score": score,
                    "label": label,
                    "pred": pred,
                }
            )
        df = pd.DataFrame(data)
        df['score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
        df["score"] = 1 - df["score"]
        
        for index, row in df.iterrows():
            if row["label"] == row["pred"]:
                df.at[index, 'score'] = 0.5 - (row['score']/2)
            else:
                df.at[index, 'score'] = (1 + row['score'])/2
            
        self.df['mistake_score'] = df['score'].values
        self.df['pred_label'] = df['pred']
        self.df['pred_label'] = self.df['pred_label'].apply(lambda x: {'label': x})
        self.df = self.df.sort_values(by='mistake_score', ascending=False)
        self.df = self.df.drop(columns=['embedding'])
        self.df.to_csv(savepath, index=False)
        return 
    