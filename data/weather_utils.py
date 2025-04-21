import torch
import numpy as np
import os
from data.weather_clustering import WeatherFeatureExtractor

class WeatherCondition:
    CLEAR = 0
    RAIN = 1
    SNOW = 2
    FOG = 3
    
    _feature_extractor = None
    _cache_dir = "cache/weather_clustering"
    
    @classmethod
    def initialize(cls, dataloader=None, force_refit=False):
        """Initialize the weather condition detector"""
        if cls._feature_extractor is None or force_refit:
            cls._feature_extractor = WeatherFeatureExtractor(model_name='resnet18', num_clusters=4)
            
            if dataloader is not None:
                cache_path = os.path.join(cls._cache_dir, "weather_clustering_model.pkl")
                cls._feature_extractor.fit(dataloader, cache_path=cache_path)
    
    @classmethod
    def get_weather_condition(cls, image_path):
        """
        Determine weather condition from image using clustering-based approach.
        This uses a pretrained CNN to extract features and then clusters them
        to identify weather patterns.
        """
        if cls._feature_extractor is None:
            raise RuntimeError("Weather condition detector not initialized. Call initialize() first.")
        
        from PIL import Image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((288, 800)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).cuda()
            
            weather_condition = cls._feature_extractor.predict_weather(image_tensor)
            return weather_condition
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.random.randint(0, 4)

    @staticmethod
    def get_weather_embedding(weather_condition):
        """
        Convert weather condition to one-hot encoding
        """
        one_hot = torch.zeros(4)
        one_hot[weather_condition] = 1
        return one_hot
    
    @classmethod
    def analyze_weather_clusters(cls, dataloader, num_samples=5):
        """
        Analyze the weather clusters by showing sample images from each cluster.
        This helps understand what kind of weather patterns the clustering has discovered.
        """
        if cls._feature_extractor is None:
            raise RuntimeError("Weather condition detector not initialized. Call initialize() first.")
        
        return cls._feature_extractor.analyze_clusters(dataloader, num_samples) 