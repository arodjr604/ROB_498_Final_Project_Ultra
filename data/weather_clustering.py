import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import pickle
from tqdm import tqdm
import torchvision.models as models

class WeatherFeatureExtractor(nn.Module):
    def __init__(self, model_name='resnet18', num_clusters=4):
        super(WeatherFeatureExtractor, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 512
        else:
            raise NotImplementedError(f"Model {model_name} not implemented")
        
        self.num_clusters = num_clusters
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        self.is_fitted = False
        
    def extract_features(self, image):
        """Extract features from an image using the pretrained model"""
        with torch.no_grad():
            features = self.model(image)
            features = features.squeeze()
        return features.cpu().numpy()
    
    def fit(self, dataloader, cache_path=None):
        """Fit the clustering model on a dataset"""
        if cache_path and os.path.exists(cache_path):
            print("Loading cached clustering model...")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.scaler = cached_data['scaler']
                self.kmeans = cached_data['kmeans']
                self.is_fitted = True
                return
        
        print("Extracting features for clustering...")
        all_features = []
        
        for batch in tqdm(dataloader):
            #images = batch['images']
            if isinstance(batch, dict):
                images = batch['images']
            else:
                images = batch[0]

            features = self.extract_features(images)
            all_features.append(features)
        
        all_features = np.vstack(all_features)
        
        print("Scaling features...")
        all_features_scaled = self.scaler.fit_transform(all_features)
        
        print("Fitting KMeans clustering...")
        self.kmeans.fit(all_features_scaled)
        
        self.is_fitted = True
        
        if cache_path:
            print("Saving clustering model...")
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'kmeans': self.kmeans
                }, f)
    
    def predict_weather(self, image):
        """Predict weather condition for a single image"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        features = self.extract_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        cluster = self.kmeans.predict(features_scaled)[0]
        
        return cluster
    
    def get_cluster_centers(self):
        """Get the cluster centers in the original feature space"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        return self.scaler.inverse_transform(self.kmeans.cluster_centers_)
    
    def analyze_clusters(self, dataloader, num_samples=5):
        """Analyze clusters by showing sample images from each cluster"""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        cluster_samples = {i: [] for i in range(self.num_clusters)}
        
        for batch in dataloader:
            #images = batch['images']
            if isinstance(batch, dict):
                images = batch['images']
            else:
                images = batch[0]
            features = self.extract_features(images)
            features_scaled = self.scaler.transform(features)
            clusters = self.kmeans.predict(features_scaled)
            
            for i, cluster in enumerate(clusters):
                if len(cluster_samples[cluster]) < num_samples:
                    cluster_samples[cluster].append(images[i].cpu())
        
        return cluster_samples 