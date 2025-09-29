import pandas as pd
import numpy as np
import faiss
from collections import defaultdict
from scipy.special import logsumexp
from scipy.stats import entropy
from tqdm import tqdm


class KNNLQC:
    def __init__(self, filepath, k=10):
        self.df = pd.read_csv(filepath)
        self.k = k
        self.embeddings = np.stack(self.df['embedding'].apply(lambda x: np.array(eval(x)))).astype(np.float32)
        self.labels = self.df['annotation_issue'].apply(lambda x: eval(x)['class_name']).tolist()
        self.filepaths = self.df['filepath'].values

    def _calculate_scores(self):
        index = faiss.IndexFlatL2(self.embeddings.shape[1])
        index.add(self.embeddings)
        
        scores = []
        for i in tqdm(range(len(self.labels))):
            distances, indices = index.search(self.embeddings[i:i+1], self.k)
            neighbor_labels = [self.labels[idx] for idx in indices[0]]
            probs = self._k_nearest_interpolation(distances[0], neighbor_labels)
            most_likely_label = max(probs, key=probs.get)
            ent = entropy(list(probs.values()))
            scores.append((most_likely_label, ent))
        
        return scores

    def _k_nearest_interpolation(self, distances, labels):
        scores = defaultdict(list)
        all_scores = [-d for d in distances]

        for d, l in zip(all_scores, labels):
            scores[l].append(d)

        denom = logsumexp(all_scores)
        probs = {l: np.exp(logsumexp(scores[l]) - denom) for l in set(labels)}
        return probs

    def save_dataframe(self, savepath):
        self.scores = self._calculate_scores()
        
        results = []
        for i, (pred, score) in enumerate(self.scores):
            results.append({
                'filepath': self.filepaths[i],
                'label': self.labels[i],
                'pred': pred,
                'score': score
            })

        df_result = pd.DataFrame(results)
        df_result['score'] = (df_result['score'] - df_result['score'].min()) / (df_result['score'].max() - df_result['score'].min())
        df_result['score'] = 1 - df_result['score']
        
        for index, row in df_result.iterrows():
            if row["label"] == row["pred"]:
                df_result.at[index, 'score'] = 0.5 - (row['score']/2)
            else:
                df_result.at[index, 'score'] = (1 + row['score'])/2
  
        # Add columns to original dataframe
        self.df['mistake_score'] = df_result['score'].values
        self.df['pred_label'] = df_result['pred']
        self.df['pred_label'] = self.df['pred_label'].apply(lambda x: {'label': x})
        self.df = self.df.sort_values(by='mistake_score', ascending=False)
        self.df = self.df.drop(columns=['embedding'])
        self.df.to_csv(savepath, index=False)
        return 
