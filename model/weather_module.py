import torch
import torch.nn as nn

class WeatherConditionModule(nn.Module):
    def __init__(self, num_conditions=4, embedding_dim=128):
        super(WeatherConditionModule, self).__init__()
        self.num_conditions = num_conditions
        self.embedding_dim = embedding_dim
        
        self.weather_embedding = nn.Embedding(num_conditions, embedding_dim)
        
        self.weather_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, weather_condition):
        weather_emb = self.weather_embedding(weather_condition)
        weather_features = self.weather_mlp(weather_emb)
        return weather_features 