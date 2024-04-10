import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN1D(nn.Module):
    def __init__(self, seq_len, dim):
        super(CNN1D, self).__init__()
        self.seq_len = seq_len
        self.dim = dim
        # Feature Extractor (1D-CNN)
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(dim, 10, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(10, 10, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(10, 1, kernel_size=10, stride=1, padding=1),
            nn.ReLU()
        )
        # RUL Regressor
        self.rul_regressor = nn.Sequential(
            nn.Linear(9, 50),
            nn.ReLU(),
            nn.Linear(50, 1),
            nn.Sigmoid()
        )
        # Domain Discriminator
        self.domain_discriminator = nn.Sequential(
            nn.Linear(9, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Feature extraction
        x = x.permute(0, 2, 1)  # Adjust dimensions for Conv1d
    
        x = self.feature_extractor(x)
       
        # Flatten the output for RUL regressor
        
        # RUL prediction
        rul_output = self.rul_regressor(x)
        rul_output = rul_output.squeeze(1)
        
        
        
        return rul_output



class CustomCNN(nn.Module):
    def __init__(self, input_channels, window_length):
        super(CustomCNN, self).__init__()
        # Assuming input_shape is (batch_size, window_length, input_channels)
        self.fe = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fe(x)
        x = x.view(x.size(0), -1) # Flatten the output for the dense layer
        x = self.regressor(x)
        return x

