import torch
import torch.nn as nn
# from .utils import common_layers
from mtr.models.utils import polyline_encoder
# from polyline_encoder import PointNetPolylineEncoder

class LidarEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        in_channels = 3
        hidden_dim = 256  

        # Initialize PointNetPolylineEncoder
        self.pointnet_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=3,  # Adjust based on your config
            num_pre_layers=3,  # Adjust based on your config
            out_channels=256  # You can set this if needed
        )

    def forward(self, lidar_input):
        # Preprocess lidar_input to create polylines (reshape, sample points, etc.)
        # lidar_input shape: (batch_size, num_points, num_features)
        # Create polylines based on the provided configuration
        
        # Assuming polylines is the processed representation of lidar_input
        # polylines = ...  # Process lidar_input to obtain polylines

        # Generate a mask to indicate valid points in the polylines
        polylines_mask = torch.ones_like(lidar_input, dtype=torch.bool)  # Example mask (all points are valid)

        # Pass the polylines and mask through the PointNetPolylineEncoder
        encoded_features = self.pointnet_encoder(lidar_input, polylines_mask)

        return encoded_features

