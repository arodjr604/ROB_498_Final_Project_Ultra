import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.weather_utils import WeatherCondition
#from utils.common import get_train_loader, merge_config
from utils.common import merge_config
from data.dataloader import get_train_loader

def visualize_clusters(cluster_samples, save_dir="results/weather_clusters"):
    """Visualize samples from each weather cluster"""
    os.makedirs(save_dir, exist_ok=True)
    
    num_clusters = len(cluster_samples)
    fig, axes = plt.subplots(num_clusters, 5, figsize=(15, 3*num_clusters))
    
    def denormalize(tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return tensor * std + mean
    
    for cluster_idx, samples in cluster_samples.items():
        for sample_idx, sample in enumerate(samples):
            if sample_idx >= 5:
                break
                
            img = denormalize(sample).clamp(0, 1)
            img = img.permute(1, 2, 0).numpy()
            
            axes[cluster_idx, sample_idx].imshow(img)
            axes[cluster_idx, sample_idx].axis('off')
            
            if sample_idx == 0:
                weather_types = ["Clear", "Rain", "Snow", "Fog"]
                axes[cluster_idx, sample_idx].set_title(f"Cluster {cluster_idx}: {weather_types[cluster_idx]}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "weather_clusters.png"))
    print(f"Cluster visualization saved to {os.path.join(save_dir, 'weather_clusters.png')}")

def main():
    args, cfg = merge_config()
    cfg.distributed = False
    cfg.data_root = '' #change this to the root directory with your CULane in it ej: '/home/arodjr/CULane/'
    
    train_dataset, train_loader, cls_num_per_lane = get_train_loader(
        cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, cfg.distributed, cfg.num_lanes
    )
    
    print("Initializing weather condition detector...")
    weather_condition = WeatherCondition()
    weather_condition.initialize(train_dataset, force_refit=True)
    
    print("Analyzing weather clusters...")
    cluster_samples = weather_condition.analyze_weather_clusters(train_dataset, num_samples=5)
    
    print("Visualizing weather clusters...")
    visualize_clusters(cluster_samples)
    
    print("Weather cluster analysis complete!")

if __name__ == "__main__":
    main() 