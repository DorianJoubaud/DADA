import torch
import numpy as np
import os
import wandb
from model import CNN1D, Discriminator, Regressor
from loss import advLoss
import itertools

class Trainer:
    def __init__(self, model,regressor, discriminator, model_optimizer, model_scheduler, print_every, epochs=200, device='cpu', prefix='FD001_FD002', wandb=False, early_rul=130):
        self.model = model.to(device)
        self.regressor = regressor.to(device)
        self.discriminator = discriminator.to(device)
        self.model_optimizer = model_optimizer
        self.model_scheduler = model_scheduler
        self.print_every = print_every
        self.epochs = epochs
        self.device = device
        self.criterion = torch.nn.MSELoss().to(device)
        self.prefix = prefix
        self.wandb = wandb
        self.early_rul = early_rul

    def train_single_epoch(self, source_dataloader, target_data_loader, epoch):
        
        alpha = 0.8
        beta = 1.0
        
        running_loss = 0
        loss_reg = 0
        loss_disc = 0
        c = 0
        
        length = min(len(source_dataloader), len(target_data_loader))
        
        source_iterator = iter(source_dataloader)
        target_iterator = iter(target_data_loader)
        
        
        for _ in range(length):
            # iterate over both dataloaders
            source_data, target_data = next(source_iterator), next(target_iterator)
            source_inputs, source_labels = source_data
            target_inputs, target_labels = target_data
            source_labels, target_labels = source_labels.unsqueeze(1), target_labels.unsqueeze(1)
            
            # move data to device
            source_inputs, source_labels = source_inputs.to(self.device), source_labels.to(self.device)
            target_inputs = target_inputs.to(self.device)
            
            # Regression loss
            source_features = self.model(source_inputs)
            target_features = self.model(target_inputs)
            
            source_predictions = self.regressor(source_features)
            
            
            loss_r = self.criterion(source_predictions, source_labels)
            loss_reg += loss_r.item()
            c += 1
            
            # Discriminator loss
            
            source_domain = self.discriminator(source_features)
            target_domain = self.discriminator(target_features)
            
            loss_d = advLoss(source_domain, target_domain, self.device)
            loss_disc += loss_d.item()
            
            
            # total loss
            if epoch < -2:
                loss = loss_r
                
            else:
                loss = alpha * loss_r + beta * loss_d
            self.model_optimizer.zero_grad()
            loss.backward()
            running_loss += loss.item()
            
            if epoch <= -2:
                torch.nn.utils.clip_grad_norm_(itertools.chain(self.model.parameters(), self.regressor.parameters()), 6)
                
            else:
                torch.nn.utils.clip_grad_norm_(itertools.chain(self.model.parameters(), self.regressor.parameters(), self.discriminator.parameters()), 6)  
            self.model_optimizer.step()
            
            
        return running_loss/length, loss_reg/c, loss_disc/c
            
            
            

        # for batch_index, data in enumerate(source_dataloader, 0):
        #     inputs, labels = data
            
        #     labels = labels.unsqueeze(1)
        #     inputs, labels = inputs.to(self.device), labels.to(self.device)
        #     self.model_optimizer.zero_grad()
        #     predictions = self.model(inputs)
          
        #     loss = self.criterion(predictions, labels)
        #     running_loss += loss.item()
        #     loss.backward()
        #     self.model_optimizer.step()
        # return running_loss/length
        
            
        
        
    
            
            

            
               

    def train(self, source_train_loader, source_val_loader, target_train_loader, target_val_loader):
        for epoch in range(self.epochs):
            
            
            
            self.model.train()
            loss, loss_r, loss_d = self.train_single_epoch(source_train_loader, target_train_loader, epoch)
            
            
            
            source_current_score, source_current_RMSE = self.test(source_val_loader)
            
            target_current_score, target_current_RMSE = self.test(target_val_loader)
            self.model_scheduler.step()
            if self.wandb:
                wandb.log({'source_score': source_current_score, 'source_RMSE': source_current_RMSE, 'target_score': target_current_score, 'target_RMSE': target_current_RMSE, 'epoch': epoch + 1,'learning_rate':self.model_scheduler.get_last_lr()[0], 'loss': loss, 'loss_r': loss_r, 'loss_d': loss_d})
            if epoch == 0:
                best_score = source_current_score
                best_RMSE = source_current_RMSE
                #print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE}')
            else:
                if source_current_score < best_score:
                    best_score = source_current_score
                    self.save_checkpoints(epoch + 1, 'best_score')
                    #print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE} (saved best_score)')
                elif source_current_RMSE < best_RMSE:
                    best_RMSE = source_current_RMSE
                    self.save_checkpoints(epoch + 1, 'best_RMSE')
                    #print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE} (saved best_rmse)')
                
                    #print(f'| Epoch {epoch + 1} | Source Score: {source_current_score} Target Score: {target_current_score} Source RMSE: {source_current_RMSE} Target RMSE: {target_current_RMSE}')
            print(f'| Epoch {epoch + 1} ')
                    
                    
            
           
        return float(best_score), float(best_RMSE)

    def save_checkpoints(self, epoch, which_type):
        state = {
            'epoch': epoch,
            'fe_state_dict': self.model.state_dict(),
            'fe_optim_dict': self.model_optimizer.state_dict(),
            'reg_state_dict': self.regressor.state_dict(),
            'reg_optim_dict': self.model_optimizer.state_dict(),
            'dis_state_dict': self.discriminator.state_dict(),
            'dis_optim_dict': self.model_optimizer.state_dict()
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
                predictions = self.regressor(self.model(inputs))
                
                                    
                score += self.score(labels * self.early_rul, predictions * self.early_rul)
                loss += criterion(labels * self.early_rul, predictions * self.early_rul) * len(labels)
        loss = (loss / len(test_loader.dataset)) ** 0.5
        score = score / len(test_loader.dataset)
        
        return score.item(), loss


