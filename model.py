import torch
import torch.nn as nn
import torch.nn.functional as F





class CNN1D(nn.Module):
    def __init__(self, input_channels, dropout=0.1):
        super(CNN1D, self).__init__()
        # Assuming input_shape is (batch_size, window_length, input_channels)
        self.dropout = nn.Dropout(dropout)
        
        self.fe = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=10, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
            
        )
    
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.fe(x)
        
        x = self.dropout(x)
        
        
        
        return x

class ReverseLayer(torch.autograd.Function):
    # Inspired from https://github.com/TL-UESTC/Domain-Adaptive-Remaining-Useful-Life-Prediction-with-Transformer
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None
    
class Discriminator(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.D = nn.Sequential(
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = self.D(x)
        x = self.dropout(x)
        return x
    
    
class Regressor(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        self.R = nn.Sequential(
            # set bias to zero and sample weight from uniform distribution between -0.1 and 0.1
            
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            
        )

        self.R2 = nn.Linear(64, 1)   
        self.init_weights()
    
    def init_weights(self) -> None:
        initrange = 0.1
        self.R2.bias.data.zero_()
        self.R2.weight.data.uniform_(-initrange, initrange)
       
        
    def forward(self, x):
        x = self.R(x)
        x = self.R2(x)
        x = nn.Sigmoid()(x)
        return x

