import ast
import rasterio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

class SimpleNowcaster(nn.Module):
    """
    A lightweight Convolutional Neural Network for precipitation nowcasting.
    Maps a stacked spatial tensor of shape (48, 81, 81) to a target (1, 41, 41).
    """
    def __init__(self, in_channels=48):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.predictor = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        features = self.encoder(x)
        downsampled = self.downsample(features)
        out = self.predictor(downsampled)
        return torch.relu(out)

class NowcastDataset(Dataset):
    """
    PyTorch Dataset for loading satellite imagery (.tif) into sequences.
    """
    def __init__(self, csv_file, root_dir, satellite='himawari', limit=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        
        # Filter anomalies to ensure exactly 3 time-steps exist
        self.df['num_files'] = self.df['last_30_minutes_observation_filename'].apply(
            lambda x: len(ast.literal_eval(x))
        )
        self.df = self.df[(self.df['satellite_target'] == satellite) & (self.df['num_files'] == 3)]
        
        if limit is not None:
            self.df = self.df.head(limit)
            
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        row = self.df.iloc[idx]
        sat_target = row['satellite_target']
        input_files = ast.literal_eval(row['last_30_minutes_observation_filename'])
        
        input_tensors = []
        for fname in input_files:
            img_path = self.root_dir / sat_target / fname
            with rasterio.open(img_path) as src:
                img = src.read().astype(np.float32) / 255.0
                input_tensors.append(img)
                
        x = np.concatenate(input_tensors, axis=0)
        x = torch.from_numpy(x)
        
        target_path = self.root_dir / 'gpm_imerg' / row['gpm_imerg_filename']
        with rasterio.open(target_path) as src:
            y = torch.from_numpy(src.read().astype(np.float32))
            
        return x, y

class PrecipitationNowcaster:
    """
    High-level interface for training and evaluating the Precipitation Nowcasting model.
    Designed for integration with solafune_tools.
    """
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
        else:
            self.device = torch.device(device)
            
        self.model = SimpleNowcaster().to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, csv_file, root_dir, epochs=2, batch_size=8, limit=None):
        """
        Trains the nowcasting model on the provided dataset.
        """
        print(f"Initializing dataset from {csv_file}...")
        dataset = NowcastDataset(csv_file, root_dir, limit=limit)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                
            avg_loss = running_loss / len(loader)
            print(f"Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}")
            
    def predict(self, input_tensor):
        """
        Predicts precipitation given an input sequence of satellite images.
        """
        self.model.eval()
        with torch.no_grad():
            input_tensor = input_tensor.to(self.device)
            if len(input_tensor.shape) == 3:
                input_tensor = input_tensor.unsqueeze(0)
            return self.model(input_tensor)
