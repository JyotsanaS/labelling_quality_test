import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import matplotlib.pyplot as plt


class SimiFeatLQC:
    def __init__(self, json_path):
        self.json_path = json_path
        self.embeddings, self.labels, self.filepaths = self.read_json()
        self.label_map = {label: i for i, label in enumerate(set(self.labels))}
        self.label_map_r = {v: k for k, v in self.label_map.items()}
        self.labels = np.array([self.label_map[label] for label in self.labels])
        self.indexes = np.array(range(len(self.labels)))
        self.num_classes = len(set(self.labels))
        self.num_data = len(self.labels)
        self.results = self.get_results()
        self.dataframe = self.get_dataframe()
        
        
    def get_dataframe(self):
        scores = self.results
        data = []
        for index, pred, score in scores:
            label = self.labels[index]
            filepath = self.filepaths[index]
            data.append({'filepath': filepath, 'label': self.label_map_r[label], 'pred': self.label_map_r[pred], 'score': score})
            
        issue_indices = [s[0] for s in scores]
        for i in range(self.num_data):
            if i not in issue_indices:
                data.append({'filepath': self.filepaths[i], 'label': self.label_map_r[self.labels[i]], 'pred': self.label_map_r[self.labels[i]], 'score': 0})   
        df = pd.DataFrame(data)
        df = df.sort_values("score", ascending=False)
        return df
    
    
    def save_score_csv(self):
        df = self.dataframe
        df.to_csv('scores.csv', index=False)
        
        
    def get_results(self, batch_size=1000, num_epoch=51):
        sel_noisy_rec = np.zeros((num_epoch, self.num_data))
        sel_times_rec = np.zeros(self.num_data)
        suggest_label_rec = np.zeros((self.num_data, self.num_classes))

        for epoch in tqdm(range(num_epoch)):
            # Process data in batches
            for batch_start in range(0, self.num_data, batch_size):
                batch_end = min(batch_start + batch_size, self.num_data)
                sel_noisy, sel_idx, suggest_label = self.simi_feat_batch(batch_start, batch_end)
                if len(sel_noisy) == 0:
                    continue
                sel_noisy_rec[epoch][np.asarray(sel_noisy)] = 1
                sel_times_rec[np.asarray(sel_idx)] += 1
                suggest_label_rec[np.asarray(sel_noisy), suggest_label] += 1

        noisy_avg = (np.sum(sel_noisy_rec, 0) + 1) / (sel_times_rec + 2)
        sel_noisy_summary = np.round(noisy_avg).astype(bool)
        num_label_errors = np.sum(sel_noisy_summary)
        print(f'[SimiFeat] We find {num_label_errors} corrupted instances from {sel_noisy_summary.shape[0]} instances')
        idx = np.argsort(noisy_avg)[-num_label_errors:][::-1] # raw index
        suggest_matrix = (suggest_label_rec + 1) / (np.sum(suggest_label_rec, 1).reshape(-1,1) + self.num_classes) # #samples * #classes

        suggest_matrix[range(len(suggest_matrix)), np.array(self.labels)] = -1
        curation = [[i, np.argmax(suggest_matrix[i]), suggest_matrix[i][np.argmax(suggest_matrix[i])] * noisy_avg[i]] for i in idx]
        return curation
        
          
    def read_json(self):
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        embeddings = np.array([e['embedding'] for e in data])
        labels = [e['label'] for e in data]
        filepaths = [e['filepath'] for e in data]
        return embeddings, labels, filepaths
        
    def cosine_distance(self, features):
        # features: N*M matrix. N features, each features is M-dimension.
        features = F.normalize(features, dim=1) # each feature's l2-norm should be 1 
        similarity_matrix = torch.matmul(features, features.T)
        distance_matrix = 1.0 - similarity_matrix
        return distance_matrix

    def get_consensus_patterns(self, sample, k=10):
        """ KNN estimation
        Args:
            sample: the index of samples
            k : the number of neighbors
        """
        feature = self.embeddings if isinstance(
            self.embeddings, torch.Tensor) else torch.tensor(self.embeddings)
        label = self.labels if isinstance(
            self.labels, torch.Tensor) else torch.tensor(self.labels)
        feature = feature[sample]
        label = label[sample]
        dist = self.cosine_distance(feature.float())
        values, indices = dist.topk(k, dim=1, largest=False, sorted=True)
        knn_labels = label[indices]
        return knn_labels, values

    def count_knn_distribution(self, sample, norm='l2'):
        """ Count the distribution of KNN
        Args:
            sample: the index of samples
        """
        num_classes = self.num_classes
        knn_labels, values = self.get_consensus_patterns(sample)
        # make the self-value less dominant (intuitive)
        values[:, 0] = 2.0 * values[:, 1] - values[:, 2]
        knn_labels_cnt = torch.zeros(len(sample), num_classes)
        for i in range(num_classes):
            knn_labels_cnt[:, i] += torch.sum((1.0 - values) * (knn_labels == i), 1)

        if norm == 'l2':
            # normalized by l2-norm -- cosine distance
            knn_labels_prob = F.normalize(knn_labels_cnt, p=2.0, dim=1)
        elif norm == 'l1':
            # normalized by mean
            knn_labels_prob = knn_labels_cnt / \
                torch.sum(knn_labels_cnt, 1).reshape(-1, 1)
        else:
            raise NameError('Undefined norm')
        return knn_labels_prob


    def get_score(self, knn_labels_cnt, label):
        """ Get the corruption score. Lower score indicates the sample is more likely to be corrupted.
        Args:
            knn_labels_cnt: KNN labels
            label: corrupted labels
        """
        score = F.nll_loss(torch.log(knn_labels_cnt + 1e-8),
                        label, reduction='none')
        return score


    def simi_feat_batch(self, start, end, method='majority_voting'):
        # Adjust function to accept start and end indices for batch processing
        idx = np.random.choice(range(start, end), int((end - start) * 0.9), replace=False)
        knn_labels_cnt = self.count_knn_distribution(sample=idx)

        score = self.get_score(knn_labels_cnt, torch.tensor(self.labels[idx]))
        score_np = score.cpu().numpy()
        sel_idx = self.indexes[idx]  # raw index


        label_pred = np.argmax(knn_labels_cnt.cpu().numpy(), axis=1).reshape(-1)
        if method == 'majority_voting':
            # test majority voting
            sel_true_false = label_pred != self.labels[idx]
            sel_noisy = (sel_idx[sel_true_false]).tolist()
            suggest_label = label_pred[sel_true_false].tolist()
        else:
            raise NameError('Undefined method')

        # raw index, raw index, suggested true label
        return sel_noisy, sel_idx, suggest_label
    
    
    def plot_images(self, df, n_rows, n_cols, flag = 0):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))
        axes = axes.flatten()
        for i, (index, row) in enumerate(df.iterrows()):
            if i >= n_rows * n_cols:
                break
            ax = axes[i]
            img = plt.imread(row["filepath"])
            
            score = row['score'] 
            if len(img.shape) < 3:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
                if flag:
                    ax.set_title(
                        f"score: {score:.4f}\ngt: {row['label']}\npred: {row['pred']}"
                    )
                else:
                    ax.set_title(
                        f"score: {score:.4f}\ngt: {row['label']}"
                    )
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        
        
    
    # mostly labelling mismatch images
    def plot_top_issue_images(self, n_rows=4, n_cols=4, reverse=False):
        df = self.dataframe
        df = df[df['label'] != df['pred']]
        df = df.sort_values("score", ascending=reverse)
        print("labelling mismatch images")
        self.plot_images(df, n_rows, n_cols, flag=1)
        
    # good images
    def plot_top_good_images(self, n_rows=4, n_cols=4):
        df = self.dataframe
        print("issue free images")
        df = df[::-1]
        self.plot_images(df, n_rows, n_cols)

if __name__ == '__main__':
    json_path = '/home/ubuntu/LQC/dataset/mscoco/roi_images/val/val.json'
    lqc = SimiFeat(json_path)
    # lqc.plot_top_issue_images(reverse=False)
    # lqc.plot_top_issue_images(reverse=True)
    # lqc.plot_top_good_images()
    lqc.save_score_csv()