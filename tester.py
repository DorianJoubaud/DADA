import torch
import numpy as np
import os
import wandb
from model import CNN1D, Discriminator, Regressor


class Tester:
    
    def __init__(self, model, regressor, early_rul, device = 'cpu'):
        self.model = model
        self.regressor = regressor
        self.device = device
        self.early_rul = early_rul
    
    
    @staticmethod
    def score(y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        diff = y_pred - y_true
        return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))
    
    
    def test(self, dataloader):
        self.model.eval()
        total_score = 0
        total_RMSE = 0
        self.model.to(self.device)
        self.regressor.to(self.device)
        
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, labels = data
                labels = labels.unsqueeze(1)
                inputs= inputs.to(self.device)
                outputs = self.regressor(self.model(inputs))
                outputs = outputs.to('cpu')
                
                total_score += self.score(labels * self.early_rul, outputs * self.early_rul)
                total_RMSE += torch.nn.MSELoss()(labels * self.early_rul, outputs * self.early_rul) * len(labels)
        total_RMSE = (total_RMSE / len(dataloader.dataset)) ** 0.5
        total_score = total_score / len(dataloader.dataset)
        
        return total_score, total_RMSE 
    