from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import json
np.random.seed(100)

class RandomForestLQC:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)
        self.embeddings = np.stack(self.df['embedding'].apply(lambda x: np.array(eval(x)))).astype(np.float32)
        self.labels = self.df['annotation_issue'].apply(lambda x: eval(x)['class_name']).tolist()
        self.filepaths = self.df['filepath'].values
        
    def save_dataframe(self, savepath):
        clf = RandomForestClassifier(random_state=0, min_samples_split=10, n_estimators=50, class_weight='balanced').fit(self.embeddings, self.labels)
        probs = clf.predict_proba(self.embeddings)
        classes = list(clf.classes_)
        
        data = []
        for i, p in enumerate(probs):
            index = p.argmax()
            pred = classes[index]
            label = self.labels[i]
            filepath = self.filepaths[i]
            data.append({'ids': i, 'filepath': filepath, 'score': p[index], 'label': label, 'pred': pred})
            
        df = pd.DataFrame(data)
        df['score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min())
        
        for index, row in df.iterrows():
            if row["label"] == row["pred"]:
                df.at[index, 'score'] = 0.5 - (row['score']/2)
            else:
                df.at[index, 'score'] = (1 + row['score'])/2
            
        self.df['mistake_score'] = df['score']
        self.df['pred_label'] = df['pred']
        self.df['pred_label'] = self.df['pred_label'].apply(lambda x: {'label':x})
        self.df = self.df.sort_values(by='mistake_score', ascending=False)
        self.df = self.df.drop(columns=['embedding'])
        self.df.to_csv(savepath, index=False)
        return 

