import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import json

# Define the autoencoder neural network model
class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        # Encoder part of the network
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        # Decoder part of the network
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Main class to handle data loading, model training, and outlier detection
class AutoEncoderLQC:
    def __init__(self, file_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = pd.read_csv(file_path)
        
        # Convert embeddings from lists to tensors
        self.df['embedding'] = self.df['embedding'].apply(lambda x: torch.tensor(eval(x), dtype=torch.float32))

        # Ensure annotation column is a dict
        self.df['annotation_issue'] = self.df['annotation_issue'].apply(lambda x: eval(x))
        
        # Move to specified device
        print("Using device:", self.device)

    def train_model(self, autoencoder, train_loader, validation_loader, epochs=50, patience=3, lr=0.0001):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(autoencoder.parameters(), lr=lr)
        best_loss = float('inf')
        epochs_no_improve = 0
        
        for epoch in range(epochs):
            autoencoder.train()
            for data in train_loader:
                optimizer.zero_grad()
                _, decoded = autoencoder(data)
                loss = criterion(decoded, data)
                loss.backward()
                optimizer.step()
            
            val_loss = self.validate(autoencoder, validation_loader, criterion)
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f'Early stopping triggered after {epoch+1} epochs!')
                    break

    def validate(self, autoencoder, validation_loader, criterion):
        autoencoder.eval()
        val_loss = 0
        with torch.no_grad():
            for data in validation_loader:
                _, decoded = autoencoder(data)
                val_loss += criterion(decoded, data).item()
        return val_loss / len(validation_loader)

    def detect_outliers(self, autoencoder, data, threshold=0.98):
        autoencoder.eval()
        with torch.no_grad():
            encoded, decoded = autoencoder(data)
            loss = torch.mean((data - decoded)**2, axis=1)
            outliers = torch.where(loss > threshold)[0]
            return outliers, loss, encoded

    def save_dataframe(self, save_path, threshold=0.98):
        self.df['mistake_score'] = 0  # Initialize a new column for mistake scores
        # Extract unique labels
        labels = self.df['annotation_issue'].apply(lambda x: x['class_name']).unique()
        all_outliers = []
        all_data = []

        centroids = {}
        embeddings_by_label = {}

        for label in tqdm(labels):
            sub_df = self.df[self.df['annotation_issue'].apply(lambda x: x['class_name']) == label]
            embeddings = torch.stack(list(sub_df['embedding'])).to(self.device)

            train_data, test_data = train_test_split(embeddings, test_size=0.2, random_state=42)
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True)
            validation_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)

            autoencoder = Autoencoder(embeddings.shape[1]).to(self.device)
            self.train_model(autoencoder, train_loader, validation_loader)

            outlier_indices, mse, encoded = self.detect_outliers(autoencoder, embeddings, threshold=0.98)
            inlier_indices = [i for i in range(len(embeddings)) if i not in outlier_indices]
            inliers = encoded[inlier_indices].cpu().numpy()

            # Calculate centroid for the current label using inliers
            centroid = np.mean(inliers, axis=0)
            centroids[label] = centroid
            embeddings_by_label[label] = encoded.cpu().numpy()

            for idx in range(len(sub_df)):
                sub_df.iloc[idx, self.df.columns.get_loc('mistake_score')] = mse[idx].item()
            
            all_data.append(sub_df)

        # Predict labels for outliers
        for sub_df in all_data:
            outlier_df = sub_df[sub_df['mistake_score'] > 0.98]
            embeddings = torch.stack(list(outlier_df['embedding'])).to(self.device)
            _, _, outlier_encoded = self.detect_outliers(autoencoder, embeddings, threshold=0)
            outlier_encoded = outlier_encoded.cpu().numpy()

            # Assign labels based on nearest centroid
            predicted_labels = []
            for oe in outlier_encoded:
                distances = {label: np.linalg.norm(oe - centroid) for label, centroid in centroids.items()}
                predicted_label = min(distances, key=distances.get)
                predicted_labels.append(predicted_label)

            outlier_df['pred_label'] = [{'label': label} for label in predicted_labels]

            # Update mistake_score based on whether the prediction matches the original label
            for idx in range(len(outlier_df)):
                original_label = outlier_df.iloc[idx]['annotation_issue']['class_name']
                predicted_label = outlier_df.iloc[idx]['pred_label']['label']
                if original_label == predicted_label:
                    outlier_df.iloc[idx, self.df.columns.get_loc('mistake_score')] *= 0.5
                else:
                    outlier_df.iloc[idx, self.df.columns.get_loc('mistake_score')] = 0.5 + 0.5 * outlier_df.iloc[idx, self.df.columns.get_loc('mistake_score')]
                    

        df = pd.concat(all_data)
        df = df.drop(columns=['embedding'])
        df.to_csv(save_path, index=False)
        print("Outliers, mistake scores, and predicted labels for outliers have been added and saved.")