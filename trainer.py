import torch
import numpy as np
import os
import wandb

class Trainer:
    def __init__(self, model, model_optimizer, model_scheduler, print_every, epochs=200, device='cpu', prefix='FD001_FD002', wandb=False, early_rul=130):
        self.model = model.to(device)
        self.model_optimizer = model_optimizer
        self.model_scheduler = model_scheduler
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.criterion = torch.nn.MSELoss().to(device)
        self.prefix = prefix
        self.wandb = wandb
        self.early_rul = early_rul

    def train_single_epoch(self, dataloader):
        running_loss = 0

        length = len(dataloader)

        for batch_index, data in enumerate(dataloader, 0):
            inputs, labels = data
            
            labels = labels.unsqueeze(1)
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.model_optimizer.zero_grad()
            predictions = self.model(inputs)
          
            loss = self.criterion(predictions, labels)
            running_loss += loss.item()
            loss.backward()
            self.model_optimizer.step()
        return running_loss/length
        
            
        
        
    
            
            

            
               

    def train(self, source_train_loader, source_val_loader, target_val_loader):
        for epoch in range(self.epochs):
            
            
            
            self.model.train()
            loss = self.train_single_epoch(source_train_loader)
            
            
            source_current_score, source_current_RMSE = self.test(source_val_loader)
            target_current_score, target_current_RMSE = self.test(target_val_loader)
            self.model_scheduler.step()
            if self.wandb:
                wandb.log({'source_score': source_current_score, 'source_RMSE': source_current_RMSE, 'target_score': target_current_score, 'target_RMSE': target_current_RMSE, 'epoch': epoch + 1,'learning_rate':self.model_scheduler.get_last_lr()[0]})
            if epoch == 0:
                best_score = source_current_score
                best_RMSE = source_current_RMSE
                print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE}')
                
            else:
                if source_current_score < best_score:
                    best_score = source_current_score
                    self.save_checkpoints(epoch + 1, 'best_score')
                    print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE} (saved best_score)')
                elif source_current_RMSE < best_RMSE:
                    best_RMSE = source_current_RMSE
                    self.save_checkpoints(epoch + 1, 'best_RMSE')
                    print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE} (saved best_rmse)')
                else:
                    print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE}')
                    
                    
            
           
        return float(best_score), float(best_RMSE)

    def save_checkpoints(self, epoch, which_type):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optim_dict': self.model_optimizer.state_dict()
        }
        torch.save(state, 'checkpoints/{}_{}.pth'.format(self.prefix, which_type))
        

    @staticmethod
    def score(y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        diff = y_pred - y_true
        return np.sum(np.where(diff < 0, np.exp(-diff/13)-1, np.exp(diff/10)-1))

    def test(self, test_loader):
        score = 0
        loss = 0
        self.model.eval()
        criterion = torch.nn.MSELoss()
        for batch_index, data in enumerate(test_loader, 0):
            with torch.no_grad():
                inputs, labels = data
                labels = labels.unsqueeze(1)
                
            
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                predictions = self.model(inputs)
                
                                    
                score += self.score(labels * self.early_rul, predictions * self.early_rul)
                loss += criterion(labels * self.early_rul, predictions * self.early_rul) * len(labels)
        loss = (loss / len(test_loader.dataset)) ** 0.5
        score = score / len(test_loader.dataset)
        
        return score.item(), loss


